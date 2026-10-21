import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const outDir = "output/supervisor_materials";
await fs.mkdir(outDir, { recursive: true });

const rows = [
  ["Full RS-BDNet", 35, 0.50, 0.8446791748855493, 0.9355560185588289, 0.5647703815781668, 0.7520212909062873, 0.8498598052930832, 0.7755518740840917, 0.7482340281596331, 0.7771675721774078, 55.304],
  ["ResNet only", 47, 0.55, 0.8302303415189599, 0.9182835652048285, 0.4363051155325543, 0.7251332091025009, 0.8224125254401924, 0.7255336038200191, 0.669350291622793, 0.717614306591643, 26.659083],
  ["Swin only", 33, 0.55, 0.8433998019817487, 0.9346571212120268, 0.5505084026360565, 0.7350306577097068, 0.843405551300288, 0.7659004332145196, 0.7363007441385097, 0.7684304614914813, 28.998052],
  ["No cross-attention", 32, 0.55, 0.8434922510220608, 0.9353903824414624, 0.5606238087146226, 0.7400184273855036, 0.8422560708320035, 0.7695721723433979, 0.7419227080273275, 0.7723935709257475, 55.153795],
  ["No building guidance", 18, 0.55, 0.8378050472190505, 0.9303367703020919, 0.5540769717628911, 0.7290398287866238, 0.8334046593641086, 0.7617145575539288, 0.7337780410451813, 0.764986142897342, 55.303079],
  ["No ordinal learning", 28, 0.65, 0.8441438635856455, 0.9379451654071175, 0.5511316188022816, 0.7463640911976099, 0.8361648284137929, 0.7679014259552004, 0.7385024658616226, 0.7701948851788294, 55.303652],
];

const wb = Workbook.create();
const summary = wb.worksheets.add("Summary");
const results = wb.worksheets.add("Ablation Results");
const classwise = wb.worksheets.add("Class-wise F1");
const plan = wb.worksheets.add("Statistical Test Plan");
const font = "Arial";
const navy = "#17365D";
const blue = "#2F75B5";
const teal = "#2A9D8F";
const lightBlue = "#D9EAF7";
const pale = "#F5F7FA";
const red = "#C00000";
const amber = "#FFF2CC";
const green = "#E2F0D9";

for (const s of [summary, results, classwise, plan]) {
  s.showGridLines = false;
  s.getRange("A1:Z100").format.font = { name: font, size: 10, color: "#1F2937" };
}

// Raw results and derived metrics.
results.getRange("A2:P2").values = [[
  "Variant", "Best epoch", "Threshold", "Localization F1", "No-damage F1",
  "Minor F1", "Major F1", "Destroyed F1", "Damage Macro-F1",
  "Damage Harmonic F1", "Combined score", "Parameters (M)",
  "Macro-F1 change vs. full", "Macro-F1 reduction", "Parameter reduction", "Macro-F1 rank"
]];
results.getRange("A3:L8").values = rows;
results.getRange("M3:M8").formulas = [
  ["=0"], ["=I4-$I$3"], ["=I5-$I$3"], ["=I6-$I$3"], ["=I7-$I$3"], ["=I8-$I$3"]
];
results.getRange("N3:N8").formulas = [
  ["=0"], ["=$I$3-I4"], ["=$I$3-I5"], ["=$I$3-I6"], ["=$I$3-I7"], ["=$I$3-I8"]
];
results.getRange("O3:O8").formulas = [
  ["=0"], ["=1-L4/$L$3"], ["=1-L5/$L$3"], ["=1-L6/$L$3"], ["=1-L7/$L$3"], ["=1-L8/$L$3"]
];
results.getRange("P3:P8").formulas = [
  ["=RANK.EQ(I3,$I$3:$I$8,0)"], ["=RANK.EQ(I4,$I$3:$I$8,0)"], ["=RANK.EQ(I5,$I$3:$I$8,0)"],
  ["=RANK.EQ(I6,$I$3:$I$8,0)"], ["=RANK.EQ(I7,$I$3:$I$8,0)"], ["=RANK.EQ(I8,$I$3:$I$8,0)"]
];
results.getRange("A1:P1").merge();
results.getRange("A1").values = [["xBD component ablation results"]];
results.getRange("A1:P1").format = { font: { name: font, size: 14, bold: true, color: navy }, rowHeight: 26 };
results.getRange("A2:P2").format = { fill: navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, wrapText: true, verticalAlignment: "center", horizontalAlignment: "center", rowHeight: 42 };
results.getRange("A3:P8").format.verticalAlignment = "center";
results.getRange("A3:A8").format.font = { name: font, size: 10, bold: true };
results.getRange("B3:B8").format.numberFormat = "0";
results.getRange("C3:K8").format.numberFormat = "0.0000";
results.getRange("L3:L8").format.numberFormat = "0.000";
results.getRange("M3:N8").format.numberFormat = "0.0000;[Red]-0.0000";
results.getRange("O3:O8").format.numberFormat = "0.0%";
results.getRange("P3:P8").format.numberFormat = "0";
results.getRange("A3:P3").format.fill = green;
results.getRange("A3:P8").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, bottom: { style: "thin", color: "#D9E2F3" } };
results.getRange("I3:I8").conditionalFormats.add("colorScale", { colors: ["#F4CCCC", "#FFF2CC", "#B6D7A8"], thresholds: ["min", { type: "percentile", value: 50 }, "max"] });
results.getRange("A2:P8").format.autofitColumns();
results.getRange("A3:A8").format.columnWidth = 24;
results.getRange("D2:K8").format.columnWidth = 15;
results.getRange("M2:O8").format.columnWidth = 18;
results.freezePanes.freezeRows(2);

// Class-wise view.
classwise.getRange("A1:G1").merge();
classwise.getRange("A1").values = [["Class-wise xBD performance"]];
classwise.getRange("A1:G1").format = { font: { name: font, size: 14, bold: true, color: navy }, rowHeight: 26 };
classwise.getRange("A3:G3").values = [["Variant", "Localization F1", "No damage", "Minor", "Major", "Destroyed", "Macro-F1"]];
classwise.getRange("A4:G9").formulas = rows.map((_, i) => {
  const r = i + 3;
  return [`='Ablation Results'!A${r}`, `='Ablation Results'!D${r}`, `='Ablation Results'!E${r}`, `='Ablation Results'!F${r}`, `='Ablation Results'!G${r}`, `='Ablation Results'!H${r}`, `='Ablation Results'!I${r}`];
});
classwise.getRange("A3:G3").format = { fill: navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center" };
classwise.getRange("B4:G9").format.numberFormat = "0.0000";
classwise.getRange("A4:A9").format.font = { name: font, size: 10, bold: true };
classwise.getRange("B4:G9").conditionalFormats.add("colorScale", { colors: ["#F4CCCC", "#FFF2CC", "#B6D7A8"], thresholds: ["min", { type: "percentile", value: 50 }, "max"] });
classwise.getRange("A4:G4").format.fill = green;
classwise.getRange("A3:G9").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, bottom: { style: "thin", color: "#D9E2F3" } };
classwise.getRange("A3:G9").format.autofitColumns();
classwise.getRange("A4:A9").format.columnWidth = 24;

// Summary with linked metrics and editable chart.
summary.getRange("A2:H2").merge();
summary.getRange("A2").values = [["RS-BDNet xBD ablation study"]];
summary.getRange("A2:H2").format = { font: { name: font, size: 16, bold: true, color: navy }, rowHeight: 28 };
summary.getRange("A3:H3").merge();
summary.getRange("A3").values = [["Single-run component analysis. Full RS-BDNet is the reference configuration."]];
summary.getRange("A3:H3").format = { font: { name: font, size: 10, italic: true, color: "#5B6573" }, rowHeight: 22 };

summary.getRange("A5:B8").values = [
  ["Key result", "Value"],
  ["Best Macro-F1", null],
  ["Largest observed reduction", null],
  ["Full-model parameters", null],
];
summary.getRange("B6").formulas = [["='Ablation Results'!I3"]];
summary.getRange("B7").formulas = [["=MAX('Ablation Results'!N4:N8)"]];
summary.getRange("B8").formulas = [["='Ablation Results'!L3"]];
summary.getRange("A5:B5").format = { fill: navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summary.getRange("A6:A8").format.font = { name: font, size: 10, bold: true };
summary.getRange("B6:B7").format.numberFormat = "0.0000";
summary.getRange("B8").format.numberFormat = "0.000\" M\"";
summary.getRange("A5:B8").format.borders = { preset: "outside", style: "thin", color: "#AAB7C4" };
summary.getRange("A5:B8").format.columnWidth = 25;

summary.getRange("A11:E11").values = [["Variant", "Macro-F1", "Loss vs. full", "Parameters (M)", "Parameter reduction"]];
summary.getRange("A12:E17").formulas = rows.map((_, i) => {
  const r = i + 3;
  return [`='Ablation Results'!A${r}`, `='Ablation Results'!I${r}`, `='Ablation Results'!N${r}`, `='Ablation Results'!L${r}`, `='Ablation Results'!O${r}`];
});
summary.getRange("A11:E11").format = { fill: navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };
summary.getRange("B12:C17").format.numberFormat = "0.0000";
summary.getRange("D12:D17").format.numberFormat = "0.000";
summary.getRange("E12:E17").format.numberFormat = "0.0%";
summary.getRange("A12:A17").format.font = { name: font, size: 10, bold: true };
summary.getRange("A12:E12").format.fill = green;
summary.getRange("A11:E17").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, bottom: { style: "thin", color: "#D9E2F3" } };
summary.getRange("A11:E17").format.autofitColumns();
summary.getRange("A12:A17").format.columnWidth = 24;

const chart = summary.charts.add("bar", [summary.getRange("A11:A17"), summary.getRange("C11:C17")]);
chart.title = "Damage Macro-F1 reduction relative to full RS-BDNet";
chart.titleTextStyle.fontSize = 14;
chart.titleTextStyle.typeface = font;
chart.hasLegend = false;
chart.xAxis = { axisType: "textAxis", textStyle: { typeface: font, fontSize: 10 } };
chart.yAxis = { minimumScale: 0, maximumScale: 0.055, numberFormatCode: "0.000", numberFormatSourceLinked: false, textStyle: { typeface: font, fontSize: 10 }, majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 } };
chart.series.items[0].fill = blue;
chart.setPosition("D5", "L18");

summary.getRange("A20:H20").merge();
summary.getRange("A20").values = [["Observed interpretation"]];
summary.getRange("A20:H20").format = { fill: lightBlue, font: { name: font, size: 11, bold: true, color: navy } };
summary.getRange("A21:H24").merge();
summary.getRange("A21").values = [[
  "The complete configuration achieved the highest xBD Macro-F1 (0.7756). The ResNet-only variant showed the largest decrease (0.0500), supporting complementary CNN and Transformer features. Building guidance, ordinal learning, and cross-attention produced smaller observed gains. These are single-run results and do not measure variation across random seeds."
]];
summary.getRange("A21:H24").format = { fill: pale, font: { name: font, size: 10, color: "#1F2937" }, wrapText: true, verticalAlignment: "top" };
summary.getRange("A1:L25").format.columnWidth = 13;
summary.getRange("A:A").format.columnWidth = 24;

// Future statistical test plan.
plan.getRange("A1:H1").merge();
plan.getRange("A1").values = [["Planned image-level statistical robustness analysis"]];
plan.getRange("A1:H1").format = { font: { name: font, size: 14, bold: true, color: navy }, rowHeight: 26 };
plan.getRange("A3:D3").values = [["Step", "Procedure", "Purpose", "Required output"]];
plan.getRange("A4:D8").values = [
  [1, "Run inference for the full model and every ablation on the identical test images.", "Create paired observations.", "Per-image predictions or confusion counts"],
  [2, "Store localization and four-class damage TP, FP, and FN counts for each image.", "Recalculate metrics consistently when classes are absent.", "One row per image and variant"],
  [3, "Perform 10,000 paired bootstrap resamples over test images.", "Estimate 95% confidence intervals for each metric and difference.", "Bootstrap distributions and confidence intervals"],
  [4, "Apply two-sided paired Wilcoxon signed-rank tests.", "Test whether paired image-level differences are centered at zero.", "Raw p-values"],
  [5, "Apply Holm correction across the five ablation comparisons.", "Control false positives caused by multiple comparisons.", "Adjusted p-values"],
];
plan.getRange("A3:D3").format = { fill: navy, font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center" };
plan.getRange("A4:D8").format = { wrapText: true, verticalAlignment: "top" };
plan.getRange("A4:A8").format = { fill: lightBlue, font: { name: font, size: 10, bold: true, color: navy }, horizontalAlignment: "center" };
plan.getRange("A3:D8").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, bottom: { style: "thin", color: "#D9E2F3" } };
plan.getRange("A:A").format.columnWidth = 7;
plan.getRange("B:B").format.columnWidth = 42;
plan.getRange("C:C").format.columnWidth = 38;
plan.getRange("D:D").format.columnWidth = 32;
plan.getRange("A4:D8").format.rowHeight = 52;
plan.getRange("A10:D10").merge();
plan.getRange("A10").values = [["Interpretation boundary"]];
plan.getRange("A10:D10").format = { fill: amber, font: { name: font, size: 11, bold: true, color: "#7F6000" } };
plan.getRange("A11:D13").merge();
plan.getRange("A11").values = [[
  "This analysis measures uncertainty across test-image composition. It does not measure training variability from random initialization, minibatch order, or stochastic augmentation. Multiple-seed experiments are a separate extension."
]];
plan.getRange("A11:D13").format = { fill: "#FFF9E6", font: { name: font, size: 10 }, wrapText: true, verticalAlignment: "top" };

wb.recalculate();

// Compact checks and previews.
const inspect = await wb.inspect({ kind: "sheet,formula,table", maxChars: 8000, tableMaxRows: 10, tableMaxCols: 16 });
await fs.writeFile(`${outDir}/ablation_workbook_inspection.txt`, inspect.ndjson);
for (const name of ["Summary", "Ablation Results", "Statistical Test Plan"]) {
  const preview = await wb.render({ sheetName: name, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${outDir}/${name.toLowerCase().replaceAll(" ", "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(`${outDir}/RSBDNet_xBD_ablation_results.xlsx`);

