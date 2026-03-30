from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List
import json

import numpy as np
import pandas as pd
import sklearn.metrics as sm
from PIL import Image

from utils import parsers, experiment_manager
from utils.experiment_manager import CfgNode
from utils.datasets_idabd import IdaBDDataset


class PathHandler:
    def __init__(self, cfg: CfgNode, subset: str, img_id: str, disaster_name: str):
        self.cfg = cfg
        self.patch = (disaster_name, img_id, subset)

        self.pred_dir = Path(cfg.PATHS.OUTPUT) / "predictions" / cfg.NAME / f"submission_{cfg.NAME}"
        self.lp = self.pred_dir / f"{disaster_name}_{img_id}_localization_disaster_prediction.png"
        self.dp = self.pred_dir / f"{disaster_name}_{img_id}_damage_disaster_prediction.png"

        self.dataset_dir = Path(cfg.PATHS.DATASET)
        self.lt = self.dataset_dir / subset / "targets" / f"{disaster_name}_{img_id}_pre_disaster_target.png"
        self.dt = self.dataset_dir / subset / "targets" / f"{disaster_name}_{img_id}_post_disaster_target.png"

        self.paths = (self.lp, self.dp, self.lt, self.dt)

    def load_and_validate_image(self, path):
        assert path.is_file(), f"file '{path}' does not exist or is not a file"
        img = np.array(Image.open(path))
        assert img.dtype == np.uint8, f"{path.name} is of wrong format {img.dtype} - should be np.uint8"
        assert img.shape == (1024, 1024), f"{path} must be a 1024x1024 image"
        return img

    def load_images(self):
        return [self.load_and_validate_image(path) for path in self.paths]


class RowPairCalculator:
    @staticmethod
    def extract_buildings(x: np.ndarray):
        buildings = x.copy()
        buildings[x > 0] = 1
        return buildings

    @staticmethod
    def compute_tp_fn_fp(pred: np.ndarray, targ: np.ndarray, c: int) -> List[int]:
        TP = np.logical_and(pred == c, targ == c).sum()
        FN = np.logical_and(pred != c, targ == c).sum()
        FP = np.logical_and(pred == c, targ != c).sum()
        return [TP, FN, FP]

    @classmethod
    def get_row_pair(cls, ph: PathHandler):
        lp, dp, lt, dt = ph.load_images()

        if getattr(ph.cfg.INFERENCE, "REGRESSION", False):
            dp = dp / 255
            dp_conv = np.zeros_like(dp)
            thresholds = ph.cfg.INFERENCE.DAMAGE_THRESHOLDS
            for i in range(1, 5):
                low_thr, up_thr = thresholds[i - 1], thresholds[i]
                d_bool = np.logical_and(low_thr < dp, dp <= up_thr)
                dp_conv[d_bool] = i
            dp = dp_conv.copy()

        lp_b, lt_b, dt_b = map(cls.extract_buildings, (lp, lt, dt))

        dp = dp * lp_b
        dp, dt = dp[dt_b == 1], dt[dt_b == 1]

        lrow = cls.compute_tp_fn_fp(lp_b, lt_b, 1)
        drow = []
        for i in range(1, 5):
            drow += cls.compute_tp_fn_fp(dp, dt, i)

        conf_mat = sm.confusion_matrix(dt.flatten(), dp.flatten(), labels=[1, 2, 3, 4])
        return (lrow, drow, ph.patch), conf_mat


class F1Recorder:
    def __init__(self, TP, FP, FN, name="", return_nan=False):
        self.TP, self.FN, self.FP, self.name, self.return_nan = TP, FN, FP, name, return_nan
        self.P = self.precision()
        self.R = self.recall()
        self.f1 = self.f1()

    def __repr__(self):
        return f"{self.name} | f1: {self.f1:.4f}, precision: {self.P:.4f}, recall: {self.R:.4f}"

    def precision(self):
        if self.TP == 0:
            if self.FN == 0 and self.return_nan:
                return float("NaN")
            return 0
        return self.TP / (self.TP + self.FP)

    def recall(self):
        if self.TP == 0:
            if self.FN == 0 and self.return_nan:
                return float("NaN")
            return 0
        return self.TP / (self.TP + self.FN)

    def f1(self):
        if self.return_nan and (np.isnan(self.P) or np.isnan(self.R)):
            return float("NaN")
        if self.P == 0 or self.R == 0:
            return 0
        return (2 * self.P * self.R) / (self.P + self.R)


class IdaBDMetrics:
    def __init__(self, cfg: CfgNode, event_keyword=None):
        self.cfg = cfg
        self.samples = IdaBDDataset(cfg, run_type="test").samples
        self.event_keyword = event_keyword

        self.dmg2str = {
            1: "No damage     (1)",
            2: "Minor damage  (2)",
            3: "Major damage  (3)",
            4: "Destroyed     (4)",
        }

        self.get_path_handlers()
        self.get_type_error()
        self.get_lf1r()
        self.get_df1rs()
        self.get_per_imgs_metrics()

    def get_path_handlers(self):
        self.path_handlers = []
        for s in self.samples:
            disaster_name, img_id, subset = s["event"], s["patch_id"], s["subset"]
            if self.event_keyword is None or self.event_keyword == disaster_name:
                self.path_handlers.append(PathHandler(self.cfg, subset, img_id, disaster_name))

    def get_type_error(self):
        with Pool() as p:
            all_rows, dmg_confs = list(zip(*p.map(RowPairCalculator.get_row_pair, self.path_handlers)))

        patch_columns = ["disaster_name", "img_id", "subset"]
        self.patches = pd.DataFrame([patch for _, _, patch in all_rows], columns=patch_columns)

        self.ldf = pd.DataFrame([lrow for lrow, _, _ in all_rows], columns=["lTP", "lFN", "lFP"])

        dcolumns = ["dTP1", "dFN1", "dFP1", "dTP2", "dFN2", "dFP2", "dTP3", "dFN3", "dFP3", "dTP4", "dFN4", "dFP4"]
        self.ddf = pd.DataFrame([drow for _, drow, _ in all_rows], columns=dcolumns)

        self.dmg_conf = np.sum(dmg_confs, axis=0)

    def get_lf1r(self):
        self.lf1r = F1Recorder(self.ldf["lTP"].sum(), self.ldf["lFP"].sum(), self.ldf["lFN"].sum(), "Buildings")

    def get_df1rs(self):
        self.df1rs = []
        for i in range(1, 5):
            self.df1rs.append(F1Recorder(
                self.ddf[f"dTP{i}"].sum(),
                self.ddf[f"dFP{i}"].sum(),
                self.ddf[f"dFN{i}"].sum(),
                self.dmg2str[i]
            ))

    @property
    def lf1(self):
        return self.lf1r.f1

    @property
    def df1s(self):
        return [x.f1 for x in self.df1rs]

    @staticmethod
    def harmonic_mean(xs):
        return len(xs) / sum((x + 1e-6) ** -1 for x in xs)

    @property
    def df1(self):
        return self.harmonic_mean(self.df1s)

    @property
    def score(self):
        return 0.3 * self.lf1 + 0.7 * self.df1

    def get_metrics_for_single_img(self, row: pd.Series):
        lf1 = F1Recorder(row["lTP"], row["lFP"], row["lFN"], "Buildings", return_nan=True).f1
        df1s = []
        for i in range(1, 5):
            df1s.append(F1Recorder(row[f"dTP{i}"], row[f"dFP{i}"], row[f"dFN{i}"], self.dmg2str[i], return_nan=True).f1)

        df1s_clean = [x for x in df1s if not np.isnan(x)]
        if len(df1s_clean) == 0:
            df1 = float("NaN")
            score = float("NaN")
        else:
            df1 = self.harmonic_mean(df1s_clean)
            score = 0.3 * lf1 + 0.7 * df1

        patch_cols = ["disaster_name", "img_id", "subset"]
        df = pd.Series(pd.concat([
            pd.Series({
                "score": score,
                "localization_f1": lf1,
                "damage_f1": df1,
                "damage_f1_no_damage": df1s[0],
                "damage_f1_minor_damage": df1s[1],
                "damage_f1_major_damage": df1s[2],
                "damage_f1_destroyed": df1s[3],
            }),
            row[patch_cols].map(str),
            row.drop(patch_cols),
        ]))
        return df

    def get_per_imgs_metrics(self):
        source_df = pd.concat([self.ldf, self.ddf, self.patches], axis=1)
        self.per_img_metrics_df = source_df.apply(self.get_metrics_for_single_img, axis=1)

    @classmethod
    def get_score(cls, cfg: CfgNode):
        print(f"Calculating metrics using {cpu_count()} cpus...")
        self = cls(cfg)

        d = {
            "score": self.score,
            "damage_f1": self.df1,
            "localization_f1": self.lf1,
            "damage_f1_no_damage": self.df1s[0],
            "damage_f1_minor_damage": self.df1s[1],
            "damage_f1_major_damage": self.df1s[2],
            "damage_f1_destroyed": self.df1s[3],
        }

        scores_dir = Path(cfg.PATHS.OUTPUT) / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)

        scores_file = scores_dir / f"scores_{cfg.NAME}.json"
        with open(scores_file, "w") as json_file:
            json.dump(d, json_file, indent=2)

        print(f"Wrote summary metrics to {scores_file.name}")
        print(d)

        out_file = scores_dir / f"{scores_file.stem}.csv"
        self.per_img_metrics_df.to_csv(out_file, index=False, sep=";")


if __name__ == "__main__":
    args = parsers.argument_parser().parse_known_args()[0]
    cfg = experiment_manager.setup_cfg(args)
    IdaBDMetrics.get_score(cfg)