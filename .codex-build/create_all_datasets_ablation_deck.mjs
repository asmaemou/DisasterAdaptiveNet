import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir="/Users/asmaemouradi/Documents/GitHub/DisasterAdaptiveNet";
const skillDir="/Users/asmaemouradi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations";
const runtimePython="/Users/asmaemouradi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const buildDir=path.join(workspaceDir,".codex-build","all-ablation-deck");
const outputDir=path.join(workspaceDir,"output","supervisor_materials");
const finalPath=path.join(outputDir,"RSBDNet_all_datasets_ablation_supervisor_presentation.pptx");
await fs.mkdir(buildDir,{recursive:true}); await fs.mkdir(outputDir,{recursive:true});

const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={navy:"#17365D",blue:"#2F75B5",teal:"#2A9D8F",pale:"#F5F7FA",light:"#D9EAF7",green:"#E2F0D9",amber:"#FFF2CC",red:"#C00000",gray:"#5B6573",line:"#D9E2F3",white:"#FFFFFF",ink:"#1F2937"};
const font="Aptos";
const variants=["Full RS-BDNet","ResNet only","Swin only","No cross-attention","No building guidance","No ordinal learning"];
const mean=[0.6658,0.5709,0.6412,0.6446,0.6289,0.6500];
const loss=[0,0.0949,0.0246,0.0212,0.0369,0.0159];
const params=[55.304,26.659,28.998,55.154,55.303,55.304];
const matrix=[
 ["xBD",0.7756,0.7255,0.7659,0.7696,0.7617,0.7679],
 ["Earthquake Turkey",0.6115,0.5105,0.5833,0.5093,0.6071,0.5631],
 ["Mount Semeru",0.5966,0.4450,0.4860,0.4967,0.5730,0.5844],
 ["Texas Tornadoes",0.5356,0.3749,0.5292,0.5868,0.3729,0.5476],
 ["Hurricane Delta",0.6769,0.6726,0.6724,0.6913,0.6989,0.7221],
 ["Pakistan Flooding",0.7990,0.6970,0.8108,0.8139,0.7598,0.7147],
];

function text(slide,value,left,top,width,height,size=24,color=C.ink,bold=false,align="left"){
 const box=slide.shapes.add({geometry:"textbox",position:{left,top,width,height},fill:"none",line:{style:"solid",fill:"none",width:0}}); box.text=value; box.text.style={fontFamily:font,fontSize:size,color,bold,alignment:align,verticalAlignment:"middle"}; return box;
}
function title(slide,value,number){text(slide,value,64,34,1080,54,30,C.navy,true);text(slide,String(number).padStart(2,"0"),1172,40,44,32,14,C.gray,true,"right");slide.shapes.add({geometry:"rect",position:{left:64,top:94,width:1152,height:3},fill:C.blue,line:{style:"solid",fill:C.blue,width:0}});}
function footer(slide,value="Six-dataset RS-BDNet ablation study"){text(slide,value,64,682,1152,18,11,"#7A8491",false,"right");}
function styleTable(t,highlight=false){t.borders.assign({style:"solid",fill:C.line,width:1});const b=t.cells.block({row:0,column:0,rowCount:t.rows.length,columnCount:t.columns.length});b.textStyle.fontFamily=font;b.textStyle.fontSize=14;b.textStyle.color=C.ink;b.margins={left:7,right:7,top:4,bottom:4};for(let c=0;c<t.columns.length;c++){t.getCell(0,c).fill=C.navy;t.getCell(0,c).text.style={fontFamily:font,fontSize:14,bold:true,color:C.white};}if(highlight){for(let c=0;c<t.columns.length;c++){t.getCell(1,c).fill=C.green;t.getCell(1,c).text.style={fontFamily:font,fontSize:14,bold:true,color:C.ink};}}}

{
 const s=p.slides.add();s.background.fill=C.navy;text(s,"RS-BDNet",76,84,480,50,18,"#A9CBE8",true);text(s,"Ablation results across six disaster datasets",76,150,1030,110,43,C.white,true);text(s,"xBD + five event-specific EBD datasets",76,278,760,42,24,"#DCE6F0");text(s,"36 trained configurations",76,420,390,28,17,"#A9CBE8",true);text(s,"6 datasets × 6 model variants",76,460,520,50,32,C.white,true);text(s,"Supervisor discussion",980,635,230,28,14,"#A9CBE8",false,"right");
}
{
 const s=p.slides.add();s.background.fill=C.white;title(s,"Experimental design",2);text(s,"Reference model",70,126,330,30,18,C.navy,true);text(s,"Full RS-BDNet combines ResNet-34 U-Net and Swin-Tiny branches, cross-attention, building guidance, and ordinal supervision.",70,164,420,132,20,C.ink);const t=s.tables.add({rows:7,columns:2,left:535,top:126,width:675,height:430,values:[["Configuration","Purpose"],["Full RS-BDNet","Reference architecture"],["ResNet only","Remove Transformer branch"],["Swin only","Remove CNN branch"],["No cross-attention","Remove explicit bi-temporal interaction"],["No building guidance","Remove localization-guided refinement"],["No ordinal learning","Remove severity-aware auxiliary loss"]]});styleTable(t,true);t.columns.get(0).width=230;t.columns.get(1).width=445;text(s,"Same protocol",70,338,330,30,18,C.navy,true);text(s,"Within each dataset, all variants used the same partitions, preprocessing, optimization, validation-based model selection, and held-out test evaluation.",70,376,420,140,19,C.ink);text(s,"One run per dataset–variant pair",70,575,420,34,18,"#8A5A00",true);footer(s);
}
{
 const s=p.slides.add();s.background.fill=C.white;title(s,"Damage Macro-F1 varies by dataset",3);const vals=[["Dataset",...variants],...matrix.map(r=>[r[0],...r.slice(1).map(v=>v.toFixed(4))])];const t=s.tables.add({rows:7,columns:7,left:42,top:126,width:1196,height:430,values:vals});styleTable(t,false);t.columns.get(0).width=205;for(let i=1;i<7;i++)t.columns.get(i).width=165;for(let r=1;r<7;r++){let best=1;for(let c=2;c<7;c++)if(Number(vals[r][c])>Number(vals[r][best]))best=c;t.getCell(r,best).fill=C.green;t.getCell(r,best).text.style={fontFamily:font,fontSize:14,bold:true,color:C.ink};}text(s,"The full model leads on xBD, Earthquake Turkey, and Mount Semeru. Specific ablations lead on Texas Tornadoes, Hurricane Delta, and Pakistan Flooding, showing event-dependent module effects.",55,582,1170,62,18,C.ink);footer(s);
}
{
 const s=p.slides.add();s.background.fill=C.white;title(s,"The full model has the highest mean Macro-F1",4);s.charts.add("bar",{position:{left:58,top:128,width:760,height:450},categories:["Full","ResNet","Swin","No cross-attn.","No guidance","No ordinal"],series:[{name:"Mean Macro-F1",values:mean,fill:C.blue,points:[{idx:0,fill:C.teal}]}],hasLegend:false,barOptions:{direction:"column",grouping:"clustered",gapWidth:55},yAxis:{min:0,max:0.72,majorUnit:0.1,numberFormatCode:"0.00",majorGridlines:{style:"solid",fill:"#E5E7EB",width:1},textStyle:{typeface:font,fontSize:14}},xAxis:{textStyle:{typeface:font,fontSize:12}},dataLabels:{showValue:true,position:"outEnd",textStyle:{fontSize:14,typeface:font},numberFormatCode:"0.0000"},chartFill:C.white,chartLine:{fill:C.white,width:0}});const t=s.tables.add({rows:7,columns:3,left:850,top:142,width:350,height:355,values:[["Variant","Mean loss","Full better"],...variants.map((v,i)=>[v.replace("Full RS-BDNet","Full"),loss[i].toFixed(4),[0,6,5,3,5,4][i].toString()+"/6"])]});styleTable(t,true);t.columns.get(0).width=190;t.columns.get(1).width=90;t.columns.get(2).width=70;text(s,"Removing the ResNet branch causes the largest average loss (0.0949). The full configuration outperforms that variant on all six datasets.",850,528,350,90,17,C.ink);footer(s);
}
{
 const s=p.slides.add();s.background.fill=C.white;title(s,"Accuracy–parameter tradeoff",5);const t=s.tables.add({rows:7,columns:4,left:80,top:132,width:760,height:410,values:[["Configuration","Mean Macro-F1","Parameters (M)","Mean loss"],...variants.map((v,i)=>[v,mean[i].toFixed(4),params[i].toFixed(3),loss[i].toFixed(4)])]});styleTable(t,true);t.columns.get(0).width=310;t.columns.get(1).width=160;t.columns.get(2).width=145;t.columns.get(3).width=145;text(s,"Key interpretation",900,150,280,30,19,C.navy,true);text(s,"The single-branch variants reduce parameters by roughly half, but both reduce average Macro-F1. Removing cross-attention, building guidance, or ordinal learning changes the parameter count negligibly; their value should therefore be judged by performance rather than compression.",900,194,290,270,19,C.ink);text(s,"Best compact option",900,510,260,28,17,C.navy,true);text(s,"Swin only: 28.998 M parameters, mean Macro-F1 0.6412",900,546,290,70,18,C.ink);footer(s);
}
{
 const s=p.slides.add();s.background.fill=C.white;title(s,"Preliminary cross-dataset statistical analysis",6);const t=s.tables.add({rows:4,columns:3,left:68,top:132,width:1144,height:300,values:[["Analysis","Result","Meaning"],["Friedman test","Q = 11.9048; p = 0.0361","At least one configuration differs in cross-dataset rank."],["Kendall's W","W = 0.3968","Moderate agreement in configuration ordering across datasets."],["Holm-adjusted post-hoc tests","No pair significant","The six datasets provide limited power for individual full-versus-ablation comparisons."]]});styleTable(t,false);t.columns.get(0).width=280;t.columns.get(1).width=240;t.columns.get(2).width=624;text(s,"Important limitation",72,474,250,28,18,"#8A5A00",true);text(s,"These tests treat the six datasets as paired blocks. They do not measure training variability because each configuration was trained only once per dataset.",72,510,1110,64,19,C.ink);footer(s);
}
{
 const s=p.slides.add();s.background.fill=C.white;title(s,"Recommended next statistical experiment",7);const t=s.tables.add({rows:6,columns:3,left:68,top:132,width:1144,height:390,values:[["Step","Procedure","Output"],["1","Save per-image TP, FP, and FN counts for every model and test image.","Paired image-level observations"],["2","Use 10,000 paired bootstrap resamples of identical test images.","95% confidence intervals for metrics and differences"],["3","Apply paired Wilcoxon signed-rank tests.","Raw p-values for full model versus each ablation"],["4","Apply Holm correction across the five ablation comparisons.","Adjusted p-values"],["5","Repeat selected training experiments with 3–5 random seeds.","Mean ± standard deviation and training-stability evidence"]]});styleTable(t,false);t.columns.get(0).width=85;t.columns.get(1).width=640;t.columns.get(2).width=419;for(let r=1;r<6;r++){t.getCell(r,0).fill=C.light;t.getCell(r,0).text.style={fontFamily:font,fontSize:16,bold:true,color:C.navy};}text(s,"Use both levels of evidence",72,560,330,28,18,C.navy,true);text(s,"Image-level resampling measures test-set uncertainty; multiple seeds measure stochastic training variability.",72,596,1090,45,19,C.ink);footer(s);
}

for(const [i,slide] of p.slides.items.entries()){const png=await p.export({slide,format:"png",scale:1});await fs.writeFile(path.join(buildDir,`slide-${String(i+1).padStart(2,"0")}.png`),new Uint8Array(await png.arrayBuffer()));}
const montage=await p.export({format:"webp",montage:true});await fs.writeFile(path.join(buildDir,"montage.webp"),new Uint8Array(await montage.arrayBuffer()));
const candidate=path.join(buildDir,"candidate.pptx");await(await PresentationFile.exportPptx(p)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL(path.join(skillDir,"container_tools/artifact_tool_utils.mjs")).href);
const req={explicitTotalSlideCount:7,requiredNativeTableOwnerSlides:[2,3,4,5,6,7],requiredNativeChartOwnerSlides:[4],materializeLiteralChartWorkbooks:true,nativeChartTargetApplication:"powerpoint"};
const result=await finalizePresentation({...req,workspaceDir,candidatePath:candidate,finalPath,pythonExecutable:runtimePython,integrityValidatorPath:path.join(skillDir,"container_tools/inspect_presentation_package_integrity.py"),layoutValidatorPath:path.join(skillDir,"container_tools/inspect_presentation_layout_geometry.py"),layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit",...req.requiredNativeTableOwnerSlides.flatMap(n=>["--require-native-table-slide",String(n)])],fontPolicy:{basis:"design",families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(buildDir,"validation.json")});console.log(JSON.stringify(result,null,2));
