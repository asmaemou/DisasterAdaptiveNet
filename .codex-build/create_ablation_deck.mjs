import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/asmaemouradi/Documents/GitHub/DisasterAdaptiveNet";
const skillDir = "/Users/asmaemouradi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations";
const runtimePython = "/Users/asmaemouradi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const buildDir = path.join(workspaceDir, ".codex-build", "ablation-deck");
const outputDir = path.join(workspaceDir, "output", "supervisor_materials");
const finalPath = path.join(outputDir, "RSBDNet_xBD_ablation_supervisor_presentation_v2.pptx");
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });

const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = { navy: "#17365D", blue: "#2F75B5", teal: "#2A9D8F", pale: "#F5F7FA", lightBlue: "#D9EAF7", green: "#E2F0D9", amber: "#FFF2CC", red: "#C00000", gray: "#5B6573", line: "#D9E2F3", white: "#FFFFFF", ink: "#1F2937" };
const font = "Aptos";
const variants = ["Full RS-BDNet", "ResNet only", "Swin only", "No cross-attention", "No building guidance", "No ordinal learning"];
const loc = [0.8447, 0.8302, 0.8434, 0.8435, 0.8378, 0.8441];
const macro = [0.7756, 0.7255, 0.7659, 0.7696, 0.7617, 0.7679];
const loss = [0, 0.0500, 0.0097, 0.0060, 0.0138, 0.0077];
const params = [55.304, 26.659, 28.998, 55.154, 55.303, 55.304];
const classRows = [
  ["Full RS-BDNet", "0.9356", "0.5648", "0.7520", "0.8499", "0.7756"],
  ["ResNet only", "0.9183", "0.4363", "0.7251", "0.8224", "0.7255"],
  ["Swin only", "0.9347", "0.5505", "0.7350", "0.8434", "0.7659"],
  ["No cross-attention", "0.9354", "0.5606", "0.7400", "0.8423", "0.7696"],
  ["No building guidance", "0.9303", "0.5541", "0.7290", "0.8334", "0.7617"],
  ["No ordinal learning", "0.9379", "0.5511", "0.7464", "0.8362", "0.7679"],
];

function text(slide, value, left, top, width, height, size=24, color=C.ink, bold=false, align="left") {
  const box = slide.shapes.add({ geometry: "textbox", position: { left, top, width, height }, fill: "none", line: { style: "solid", fill: "none", width: 0 } });
  box.text = value;
  box.text.style = { fontFamily: font, fontSize: size, color, bold, alignment: align, verticalAlignment: "middle" };
  return box;
}

function title(slide, value, number) {
  text(slide, value, 64, 34, 1080, 54, 30, C.navy, true);
  text(slide, String(number).padStart(2, "0"), 1172, 40, 44, 32, 14, C.gray, true, "right");
  const rule = slide.shapes.add({ geometry: "rect", position: { left: 64, top: 94, width: 1152, height: 3 }, fill: C.blue, line: { style: "solid", fill: C.blue, width: 0 } });
  return rule;
}

function footer(slide, value="xBD ablation study") {
  text(slide, value, 64, 682, 1152, 18, 11, "#7A8491", false, "right");
}

function styleTable(table, headerRows=1, highlightFull=true) {
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  const all = table.cells.block({ row: 0, column: 0, rowCount: table.rows.length, columnCount: table.columns.length });
  all.textStyle.fontFamily = font;
  all.textStyle.fontSize = 15;
  all.textStyle.color = C.ink;
  all.margins = { left: 8, right: 8, top: 5, bottom: 5 };
  for (let r=0; r<headerRows; r++) for (let c=0; c<table.columns.length; c++) {
    table.getCell(r,c).fill = C.navy;
    table.getCell(r,c).text.style = { fontFamily: font, fontSize: 15, bold: true, color: C.white };
  }
  if (highlightFull && table.rows.length > headerRows) {
    for (let c=0; c<table.columns.length; c++) {
      table.getCell(headerRows,c).fill = C.green;
      table.getCell(headerRows,c).text.style = { fontFamily: font, fontSize: 15, bold: true, color: C.ink };
    }
  }
}

// Slide 1
{
  const s = p.slides.add();
  s.background.fill = C.navy;
  text(s, "RS-BDNet", 76, 84, 480, 50, 18, "#A9CBE8", true);
  text(s, "xBD component ablation results", 76, 150, 950, 82, 46, C.white, true);
  text(s, "Observed performance, parameter tradeoffs, and planned statistical analysis", 76, 246, 900, 70, 23, "#DCE6F0");
  text(s, "Complete model Macro-F1", 76, 400, 300, 30, 17, "#A9CBE8", true);
  text(s, "0.7756", 76, 438, 300, 78, 54, C.white, true);
  text(s, "Six configurations evaluated with the same xBD protocol", 76, 590, 760, 36, 18, "#DCE6F0");
  text(s, "Supervisor discussion", 980, 635, 230, 28, 14, "#A9CBE8", false, "right");
}

// Slide 2
{
  const s = p.slides.add();
  s.background.fill = C.white;
  title(s, "Ablation design", 2);
  text(s, "Reference configuration", 72, 124, 360, 32, 18, C.navy, true);
  text(s, "Full RS-BDNet combines the ResNet-34 U-Net and Swin-Tiny branches with cross-attention, building guidance, and ordinal supervision.", 72, 162, 420, 150, 20, C.ink);
  const t = s.tables.add({ rows: 6, columns: 2, left: 550, top: 126, width: 650, height: 390, values: [
    ["Configuration", "Question answered"],
    ["ResNet only", "What is lost without the Transformer branch?"],
    ["Swin only", "What is lost without the CNN branch?"],
    ["No cross-attention", "Does explicit bi-temporal interaction help?"],
    ["No building guidance", "Does localization-guided damage prediction help?"],
    ["No ordinal learning", "Does severity-aware supervision help?"],
  ]});
  styleTable(t, 1, false);
  t.columns.get(0).width = 215;
  t.columns.get(1).width = 435;
  text(s, "Control conditions", 72, 350, 360, 32, 18, C.navy, true);
  text(s, "Each variant used the same xBD partitions, preprocessing, optimization procedure, validation-based checkpoint selection, and held-out test evaluation.", 72, 388, 420, 145, 19, C.ink);
  text(s, "Current limitation", 72, 564, 210, 28, 17, "#8A5A00", true);
  text(s, "One training run per configuration", 72, 598, 420, 34, 18, C.gray);
  footer(s);
}

// Slide 3
{
  const s = p.slides.add();
  s.background.fill = C.white;
  title(s, "The complete model achieved the highest xBD Macro-F1", 3);
  const chart = s.charts.add("bar", {
    position: { left: 66, top: 126, width: 760, height: 450 },
    categories: ["Full", "ResNet", "Swin", "No cross-attn.", "No guidance", "No ordinal"],
    series: [{ name: "Damage Macro-F1", values: macro, fill: C.blue, points: [{ idx: 0, fill: C.teal }] }],
    hasLegend: false,
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
    yAxis: { min: 0, max: 0.85, majorUnit: 0.10, numberFormatCode: "0.00", majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 }, textStyle: { typeface: font, fontSize: 14 } },
    xAxis: { textStyle: { typeface: font, fontSize: 12 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 14, typeface: font }, numberFormatCode: "0.0000" },
    chartFill: C.white,
    chartLine: { fill: C.white, width: 0 },
  });
  text(s, "Observed Macro-F1 losses", 875, 134, 300, 30, 18, C.navy, true);
  const t = s.tables.add({ rows: 6, columns: 2, left: 870, top: 176, width: 330, height: 300, values: [
    ["Removed component", "Loss"],
    ["ResNet-only model", "0.0500"],
    ["Building guidance", "0.0138"],
    ["CNN branch", "0.0097"],
    ["Ordinal learning", "0.0077"],
    ["Cross-attention", "0.0060"],
  ]});
  styleTable(t, 1, false);
  t.columns.get(0).width = 230;
  t.columns.get(1).width = 100;
  text(s, "The single-branch results support complementary CNN and Transformer features. The smaller component gains still require uncertainty analysis.", 870, 510, 330, 112, 17, C.ink);
  footer(s);
}

// Slide 4
{
  const s = p.slides.add();
  s.background.fill = C.white;
  title(s, "Performance and parameter tradeoffs", 4);
  const chart = s.charts.add("bar", {
    position: { left: 70, top: 132, width: 690, height: 420 },
    categories: ["Full", "ResNet", "Swin", "No cross-attn.", "No guidance", "No ordinal"],
    series: [{ name: "Macro-F1 loss vs. full", values: loss, fill: C.blue, points: [{ idx: 1, fill: C.red }] }],
    hasLegend: false,
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
    yAxis: { min: 0, max: 0.055, majorUnit: 0.01, numberFormatCode: "0.000", majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 }, textStyle: { typeface: font, fontSize: 14 } },
    xAxis: { textStyle: { typeface: font, fontSize: 12 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 14, typeface: font }, numberFormatCode: "0.0000" },
    chartFill: C.white,
    chartLine: { fill: C.white, width: 0 },
  });
  const t = s.tables.add({ rows: 7, columns: 3, left: 800, top: 138, width: 400, height: 365, values: [
    ["Configuration", "Params (M)", "Reduction"],
    ["Full", "55.304", "0.0%"],
    ["ResNet only", "26.659", "51.8%"],
    ["Swin only", "28.998", "47.6%"],
    ["No cross-attn.", "55.154", "0.3%"],
    ["No building guide", "55.303", "0.0%"],
    ["No ordinal", "55.304", "0.0%"],
  ]});
  styleTable(t, 1, true);
  t.columns.get(0).width = 210;
  t.columns.get(1).width = 100;
  t.columns.get(2).width = 90;
  text(s, "Parameter interpretation", 802, 536, 300, 28, 17, C.navy, true);
  text(s, "Single branches nearly halve the parameter count, but both reduce Macro-F1. Removing the smaller modules changes parameters very little.", 802, 570, 394, 80, 17, C.ink);
  footer(s);
}

// Slide 5
{
  const s = p.slides.add();
  s.background.fill = C.white;
  title(s, "Class-wise damage results", 5);
  const t = s.tables.add({ rows: 7, columns: 6, left: 72, top: 134, width: 1136, height: 410, values: [
    ["Configuration", "No damage", "Minor", "Major", "Destroyed", "Macro-F1"],
    ...classRows,
  ]});
  styleTable(t, 1, true);
  t.columns.get(0).width = 300;
  for (let i=1; i<6; i++) t.columns.get(i).width = 167;
  text(s, "Minor damage remains the weakest class for every configuration. The full model produces its largest class-wise advantage over ResNet only on minor damage (+0.1285).", 74, 574, 1130, 58, 18, C.ink);
  footer(s);
}

// Slide 6
{
  const s = p.slides.add();
  s.background.fill = C.white;
  title(s, "Planned statistical robustness analysis", 6);
  const t = s.tables.add({ rows: 6, columns: 3, left: 68, top: 132, width: 1144, height: 390, values: [
    ["Step", "Planned procedure", "What it provides"],
    ["1", "Run all saved checkpoints on the same test images.", "Paired image-level observations"],
    ["2", "Store per-image localization and damage confusion counts.", "Consistent multi-class metric calculation"],
    ["3", "Perform 10,000 paired bootstrap resamples.", "95% confidence intervals for metrics and differences"],
    ["4", "Apply two-sided paired Wilcoxon signed-rank tests.", "Raw p-values for full model versus each ablation"],
    ["5", "Apply Holm correction across five comparisons.", "Adjusted p-values with controlled false positives"],
  ]});
  styleTable(t, 1, false);
  t.columns.get(0).width = 85;
  t.columns.get(1).width = 600;
  t.columns.get(2).width = 459;
  for (let r=1; r<6; r++) {
    t.getCell(r, 0).fill = C.lightBlue;
    t.getCell(r, 0).text.style = { fontFamily: font, fontSize: 16, bold: true, color: C.navy };
  }
  text(s, "Interpretation boundary", 72, 554, 260, 28, 17, "#8A5A00", true);
  text(s, "This test measures uncertainty across test-image composition. It does not measure training variability across random seeds. Multiple-seed training remains a separate extension.", 72, 588, 1128, 64, 18, C.ink);
  footer(s, "Planned analysis based on the Insta-BDA image-level approach");
}

// Render every slide and a montage before finalization.
for (const [i, slide] of p.slides.items.entries()) {
  const png = await p.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${String(i+1).padStart(2,"0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
const montage = await p.export({ format: "webp", montage: true });
await fs.writeFile(path.join(buildDir, "montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const candidate = path.join(buildDir, "candidate.pptx");
await (await PresentationFile.exportPptx(p)).save(candidate);

const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
const requirements = {
  explicitTotalSlideCount: 6,
  requiredNativeTableOwnerSlides: [2,3,4,5,6],
  requiredNativeChartOwnerSlides: [3,4],
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
};
const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath: candidate,
  finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit", ...requirements.requiredNativeTableOwnerSlides.flatMap(n => ["--require-native-table-slide", String(n)])],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  requiredNativeChartOwnerSlides: requirements.requiredNativeChartOwnerSlides,
  materializeLiteralChartWorkbooks: true,
  nativeChartTargetApplication: "powerpoint",
  fontPolicy: { basis: "design", families: [font] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "validation-v2.json"),
});
console.log(JSON.stringify(result, null, 2));

