import fs from "node:fs/promises";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const outDir = "output/supervisor_materials";
await fs.mkdir(outDir, { recursive: true });

const xbd = `dataset,variant,best_epoch,localization_threshold,localization_f1,no_damage_f1,minor_damage_f1,major_damage_f1,destroyed_f1,damage_macro_f1,damage_harmonic_f1,combined_score,parameters_m
xBD,full,35,0.5000,0.8446791748855493,0.9355560185588289,0.5647703815781668,0.7520212909062873,0.8498598052930832,0.7755518740840917,0.7482340281596331,0.7771675721774078,55.304
xBD,resnet_only,47,0.5500,0.8302303415189599,0.9182835652048285,0.4363051155325543,0.7251332091025009,0.8224125254401924,0.7255336038200191,0.669350291622793,0.717614306591643,26.659083
xBD,swin_only,33,0.5500,0.8433998019817487,0.9346571212120268,0.5505084026360565,0.7350306577097068,0.843405551300288,0.7659004332145196,0.7363007441385097,0.7684304614914813,28.998052
xBD,no_cross_attention,32,0.5500,0.8434922510220608,0.9353903824414624,0.5606238087146226,0.7400184273855036,0.8422560708320035,0.7695721723433979,0.7419227080273275,0.7723935709257475,55.153795
xBD,no_building_guidance,18,0.5500,0.8378050472190505,0.9303367703020919,0.5540769717628911,0.7290398287866238,0.8334046593641086,0.7617145575539288,0.7337780410451813,0.764986142897342,55.303079
xBD,no_ordinal,28,0.6500,0.8441438635856455,0.9379451654071175,0.5511316188022816,0.7463640911976099,0.8361648284137929,0.7679014259552004,0.7385024658616226,0.7701948851788294,55.303652`;

const eventText = await fs.readFile("/Users/asmaemouradi/.codex/attachments/64f66065-9c8b-450c-840e-1215187c8921/Pasted text.txt", "utf8");
const parse = (text) => text.trim().split(/\r?\n/).slice(1).filter(Boolean).map(line => {
  const c = line.split(",");
  return [c[0], c[1], Number(c[2]), ...c.slice(3).map(Number)];
});
const data = [...parse(xbd), ...parse(eventText)];
const datasetLabels = {
  xBD: "xBD", earthquake_turkey: "Earthquake Turkey", mount_semeru_eruption: "Mount Semeru Eruption",
  texas_tornadoes: "Texas Tornadoes", hurricane_delta: "Hurricane Delta", pakistan_flooding: "Pakistan Flooding"
};
const variantLabels = {
  full: "Full RS-BDNet", resnet_only: "ResNet only", swin_only: "Swin only",
  no_cross_attention: "No cross-attention", no_building_guidance: "No building guidance", no_ordinal: "No ordinal learning"
};
data.forEach(r => { r[0] = datasetLabels[r[0]]; r[1] = variantLabels[r[1]]; });
// Keep only the columns intended for the supervisor-facing workbook.
// The harmonic F1 and combined validation score remain part of the original
// experiment CSVs but are intentionally omitted from this presentation file.
const displayData = data.map(r => [r[0], r[1], r[2], ...r.slice(4, 10), r[12]]);
const datasets = Object.values(datasetLabels);
const variants = Object.values(variantLabels);

const wb = Workbook.create();
const summary = wb.worksheets.add("Summary");
const all = wb.worksheets.add("All Results");
const matrix = wb.worksheets.add("Macro-F1 Matrix");
const stats = wb.worksheets.add("Statistical Analysis");
const font = "Arial", navy = "#17365D", blue = "#2F75B5", light = "#D9EAF7", green = "#E2F0D9", pale = "#F5F7FA", amber = "#FFF2CC";
for (const s of [summary, all, matrix, stats]) { s.showGridLines = false; s.getRange("A1:Z120").format.font = { name: font, size: 10, color: "#1F2937" }; }

const headers = ["Dataset","Variant","Best epoch","Localization F1","No-damage F1","Minor F1","Major F1","Destroyed F1","Damage Macro-F1","Parameters (M)"];
all.getRange("A1:J1").merge(); all.getRange("A1").values = [["RS-BDNet ablation results across all six datasets"]];
all.getRange("A1:J1").format = { font:{name:font,size:14,bold:true,color:navy}, rowHeight:26 };
all.getRange("A2:J2").values = [headers];
all.getRange("A3:J38").values = displayData;
const fullRows = new Map(); data.forEach((r,i)=>{ if(r[1]==="Full RS-BDNet") fullRows.set(r[0], i+3); });
all.getRange("A2:J2").format = { fill:navy,font:{name:font,size:10,bold:true,color:"#FFFFFF"},wrapText:true,horizontalAlignment:"center",verticalAlignment:"center",rowHeight:38 };
all.getRange("A3:B38").format.font = {name:font,size:10,bold:true};
all.getRange("C3:C38").format.numberFormat="0"; all.getRange("D3:I38").format.numberFormat="0.0000"; all.getRange("J3:J38").format.numberFormat="0.000";
for (const r of fullRows.values()) all.getRange(`A${r}:J${r}`).format.fill = green;
all.getRange("I3:I38").conditionalFormats.add("colorScale",{colors:["#F4CCCC","#FFF2CC","#B6D7A8"],thresholds:["min",{type:"percentile",value:50},"max"]});
all.getRange("A2:J38").format.borders={insideHorizontal:{style:"thin",color:"#D9E2F3"},bottom:{style:"thin",color:"#D9E2F3"}};
all.getRange("A2:J38").format.autofitColumns(); all.getRange("A:A").format.columnWidth=24; all.getRange("B:B").format.columnWidth=24; all.freezePanes.freezeRows(2);

matrix.getRange("A1:G1").merge(); matrix.getRange("A1").values=[["Damage Macro-F1 by dataset and ablation variant"]];
matrix.getRange("A1:G1").format={font:{name:font,size:14,bold:true,color:navy},rowHeight:26};
matrix.getRange("A3:G3").values=[["Dataset",...variants]];
matrix.getRange("A4:A9").values=datasets.map(d=>[d]);
matrix.getRange("B4:G9").formulas=datasets.map(d=>variants.map(v=>{
  const idx=data.findIndex(r=>r[0]===d&&r[1]===v); return `='All Results'!I${idx+3}`;
}));
matrix.getRange("A3:G3").format={fill:navy,font:{name:font,size:10,bold:true,color:"#FFFFFF"},wrapText:true,horizontalAlignment:"center",verticalAlignment:"center",rowHeight:44};
matrix.getRange("A4:A9").format.font={name:font,size:10,bold:true}; matrix.getRange("B4:G9").format.numberFormat="0.0000";
matrix.getRange("B4:G9").conditionalFormats.add("colorScale",{colors:["#F4CCCC","#FFF2CC","#B6D7A8"],thresholds:["min",{type:"percentile",value:50},"max"]});
matrix.getRange("A3:G9").format.borders={insideHorizontal:{style:"thin",color:"#D9E2F3"},bottom:{style:"thin",color:"#D9E2F3"}}; matrix.getRange("A:G").format.columnWidth=20; matrix.getRange("A:A").format.columnWidth=25;

summary.getRange("A1:H1").merge(); summary.getRange("A1").values=[["RS-BDNet six-dataset ablation summary"]]; summary.getRange("A1:H1").format={font:{name:font,size:16,bold:true,color:navy},rowHeight:28};
summary.getRange("A2:H2").merge(); summary.getRange("A2").values=[["The workbook combines xBD with five event-specific EBD datasets (36 experiment rows)."]]; summary.getRange("A2:H2").format={font:{name:font,size:10,italic:true,color:"#5B6573"},rowHeight:22};
summary.getRange("A4:D4").values=[["Variant","Mean Macro-F1","Full better (datasets)","Parameters (M)"]];
summary.getRange("A5:A10").values=variants.map(v=>[v]);
summary.getRange("B5:B10").formulas=variants.map(v=>[`=SUMIF('All Results'!$B$3:$B$38,A${variants.indexOf(v)+5},'All Results'!$I$3:$I$38)/COUNTIF('All Results'!$B$3:$B$38,A${variants.indexOf(v)+5})`]);
summary.getRange("C5:C10").values=[[0],[6],[5],[3],[5],[4]];
summary.getRange("D5:D10").formulas=variants.map(v=>{const idx=data.findIndex(r=>r[1]===v);return [`='All Results'!J${idx+3}`]});
summary.getRange("A4:D4").format={fill:navy,font:{name:font,size:10,bold:true,color:"#FFFFFF"},wrapText:true,horizontalAlignment:"center"}; summary.getRange("A5:A10").format.font={name:font,size:10,bold:true}; summary.getRange("A5:D5").format.fill=green;
summary.getRange("B5:B10").format.numberFormat="0.0000"; summary.getRange("C5:C10").format.numberFormat="0"; summary.getRange("D5:D10").format.numberFormat="0.000"; summary.getRange("A4:D10").format.borders={insideHorizontal:{style:"thin",color:"#D9E2F3"},bottom:{style:"thin",color:"#D9E2F3"}}; summary.getRange("A:D").format.columnWidth=22;
const chart=summary.charts.add("bar",[summary.getRange("A4:A10"),summary.getRange("B4:B10")]); chart.title="Mean damage Macro-F1 across six datasets"; chart.hasLegend=false; chart.series.items[0].fill=blue; chart.setPosition("F4","M17");
summary.getRange("A13:E13").merge(); summary.getRange("A13").values=[["Main findings"]]; summary.getRange("A13:E13").format={fill:light,font:{name:font,size:11,bold:true,color:navy}};
summary.getRange("A14:E18").merge(); summary.getRange("A14").values=[["Across the six datasets, the full model achieved the highest mean damage Macro-F1 (0.6658). Removing the ResNet branch caused the largest mean reduction (0.0949) and reduced performance on all six datasets. The effects of individual modules varied by event: the full model was not the best configuration on every dataset, particularly for Hurricane Delta and Pakistan Flooding. These are single-run results and should not be interpreted as evidence of training-seed stability."]]; summary.getRange("A14:E18").format={fill:pale,wrapText:true,verticalAlignment:"top"};

stats.getRange("A1:F1").merge(); stats.getRange("A1").values=[["Statistical interpretation and next analysis"]]; stats.getRange("A1:F1").format={font:{name:font,size:14,bold:true,color:navy},rowHeight:26};
stats.getRange("A3:D3").values=[["Analysis","Result","Interpretation","Limitation"]];
stats.getRange("A4:D6").values=[
  ["Friedman test across six datasets","Q = 11.9048; p = 0.0361","At least one configuration has a different cross-dataset rank.","Only six dataset-level observations; datasets are heterogeneous."],
  ["Kendall's W","W = 0.3968","Moderate agreement in the ordering of configurations across datasets.","Describes rank consistency, not training randomness."],
  ["Holm-adjusted paired post-hoc tests","No comparison significant","The omnibus result does not identify a single ablation as significantly different from the full model after correction.","Low power with n = 6 datasets."],
];
stats.getRange("A3:D3").format={fill:navy,font:{name:font,size:10,bold:true,color:"#FFFFFF"},horizontalAlignment:"center"}; stats.getRange("A4:D6").format={wrapText:true,verticalAlignment:"top",rowHeight:55}; stats.getRange("A3:D6").format.borders={insideHorizontal:{style:"thin",color:"#D9E2F3"},bottom:{style:"thin",color:"#D9E2F3"}};
stats.getRange("A8:D8").merge(); stats.getRange("A8").values=[["Recommended stronger experiment"]]; stats.getRange("A8:D8").format={fill:amber,font:{name:font,size:11,bold:true,color:"#7F6000"}};
stats.getRange("A9:D13").merge(); stats.getRange("A9").values=[["For publication-level uncertainty estimates, retain per-image TP, FP, and FN counts for the full model and every ablation. Use paired bootstrap resampling of the identical test images (for example, 10,000 resamples) to obtain 95% confidence intervals for metric differences, then apply paired Wilcoxon signed-rank tests with Holm correction. This measures sensitivity to test-image composition. To measure training variability, rerun the full model and selected ablations with at least three, preferably five, random seeds and report mean ± standard deviation."]]; stats.getRange("A9:D13").format={fill:"#FFF9E6",wrapText:true,verticalAlignment:"top"}; stats.getRange("A:D").format.columnWidth=30; stats.getRange("A:A").format.columnWidth=28; stats.getRange("B:B").format.columnWidth=22;

wb.recalculate();
const inspect=await wb.inspect({kind:"sheet,formula,table",maxChars:10000,tableMaxRows:12,tableMaxCols:14}); await fs.writeFile(`${outDir}/all_datasets_ablation_inspection.txt`,inspect.ndjson);
for(const name of ["Summary","Macro-F1 Matrix","Statistical Analysis"]){const p=await wb.render({sheetName:name,autoCrop:"all",scale:1,format:"png"});await fs.writeFile(`${outDir}/${name.toLowerCase().replaceAll(" ","_")}_all.png`,new Uint8Array(await p.arrayBuffer()));}
const out=await SpreadsheetFile.exportXlsx(wb); await out.save(`${outDir}/RSBDNet_all_datasets_ablation_results.xlsx`);
