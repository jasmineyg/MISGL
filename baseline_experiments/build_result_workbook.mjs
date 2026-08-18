import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}


function asTypedRows(rows) {
  const numeric = new Set([
    "fold", "fold_seed", "test_n", "n", "acc", "precision", "recall",
    "specificity", "f1", "f1_macro", "balanced_acc", "tn", "fp", "fn",
    "tp", "roc_auc", "pr_auc",
  ]);
  const header = rows[0];
  return [
    header,
    ...rows.slice(1).map((row) =>
      row.map((value, index) => numeric.has(header[index]) ? Number(value) : value),
    ),
  ];
}


function styleTitle(sheet, rangeAddress, title) {
  const range = sheet.getRange(rangeAddress);
  range.merge();
  range.values = [[title]];
  range.format = {
    fill: "#16324F",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 30;
}


function styleHeader(range) {
  range.format = {
    fill: "#2F6690",
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: {
      bottom: { style: "medium", color: "#16324F" },
    },
  };
  range.format.rowHeight = 32;
}


const [inputCsv, outputXlsx, previewDir] = process.argv.slice(2);
if (!inputCsv || !outputXlsx || !previewDir) {
  throw new Error(
    "usage: node build_result_workbook.mjs <all_fold_metrics.csv> <output.xlsx> <preview-dir>",
  );
}

const rawRows = parseCsv(await fs.readFile(inputCsv, "utf8"));
const foldRows = asTypedRows(rawRows);
if (foldRows.length !== 1051) {
  throw new Error(`expected 1050 fold records, found ${foldRows.length - 1}`);
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
const summary = workbook.worksheets.add("Summary");
const folds = workbook.worksheets.add("Fold Metrics");
const configs = workbook.worksheets.add("Configurations");
const validation = workbook.worksheets.add("Validation");

// README / protocol
styleTitle(readme, "A1:H1", "MISGL Baseline Comparison — Reproducible 10-fold Results");
const readmeRows = [
  ["Field", "Value"],
  ["Purpose", "Paper comparison results for four MISGL variants and three published baselines."],
  ["Protocol", "Fixed grouped-stratified 10-fold split; test fold f, validation fold (f+1) mod 10, remaining eight folds train."],
  ["Seed", 1024],
  ["Prediction rule", "Positive iff probability > 0.5; exact ties are negative."],
  ["Aggregation", "Mean ± sample standard deviation over the ten fixed test folds."],
  ["Selection", "Validation loss only; test folds are used once for final evaluation."],
  ["Expected records", 1050],
  ["Expected groups", 105],
  ["Metrics", "ACC, precision, recall, specificity, binary F1, macro F1, balanced ACC, ROC-AUC, PR-AUC, TN/FP/FN/TP."],
  ["Attention source", "Ilse et al. (2018), Attention-based Deep Multiple Instance Learning."],
  ["RGMIL source", "Zhao et al. (2024), Reinforced GNNs for Multiple Instance Learning."],
  ["SubGNN source", "Alsentzer et al. (2020), Subgraph Neural Networks."],
  ["Result provenance", "/data/yg/Subgraph-MIL/diffpool2/results/baseline_comparison_20260815/final"],
  ["Split provenance", "/data/yg/Subgraph-MIL/diffpool2/results/paper_10fold_20260814/execution_manifest.json"],
];
readme.getRange(`A3:B${readmeRows.length + 2}`).values = readmeRows;
styleHeader(readme.getRange("A3:B3"));
readme.getRange("A4:A17").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#16324F" },
};
readme.getRange("A3:B17").format.borders = {
  preset: "inside",
  style: "thin",
  color: "#CBD5E1",
};
readme.getRange("A:B").format.wrapText = true;
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 100;
readme.getRange("A4:B17").format.rowHeight = 30;
readme.showGridLines = false;
readme.freezePanes.freezeRows(3);

// Raw fold metrics
folds.getRangeByIndexes(0, 0, foldRows.length, foldRows[0].length).values = foldRows;
styleHeader(folds.getRange("A1:U1"));
folds.getRange("D2:G1051").format.numberFormat = "0";
folds.getRange("H2:N1051").format.numberFormat = "0.0000";
folds.getRange("O2:R1051").format.numberFormat = "0";
folds.getRange("S2:T1051").format.numberFormat = "0.0000";
folds.getRange("A:A").format.columnWidth = 22;
folds.getRange("B:B").format.columnWidth = 44;
folds.getRange("C:C").format.columnWidth = 24;
folds.getRange("D:G").format.columnWidth = 12;
folds.getRange("H:N").format.columnWidth = 14;
folds.getRange("O:R").format.columnWidth = 10;
folds.getRange("S:T").format.columnWidth = 14;
folds.getRange("U:U").format.columnWidth = 84;
folds.getRange("U:U").format.wrapText = false;
folds.freezePanes.freezeRows(1);
folds.freezePanes.freezeColumns(3);
folds.showGridLines = false;
const foldTable = folds.tables.add("A1:U1051", true, "FoldMetricsTable");
foldTable.showBandedColumns = false;

// Formula-backed summary, using exact contiguous ten-row fold blocks.
const metricColumns = [
  ["acc", "H"], ["precision", "I"], ["recall", "J"],
  ["specificity", "K"], ["f1", "L"], ["f1_macro", "M"],
  ["balanced_acc", "N"], ["roc_auc", "S"], ["pr_auc", "T"],
];
const groupRanges = [];
let start = 1;
while (start < foldRows.length) {
  const [datasetKey, dataName, method] = foldRows[start];
  let end = start;
  while (
    end + 1 < foldRows.length
    && foldRows[end + 1][0] === datasetKey
    && foldRows[end + 1][2] === method
  ) {
    end += 1;
  }
  groupRanges.push({ datasetKey, dataName, method, startRow: start + 1, endRow: end + 1 });
  start = end + 1;
}
if (groupRanges.length !== 105) {
  throw new Error(`expected 105 dataset-method groups, found ${groupRanges.length}`);
}

styleTitle(summary, "A1:V1", "10-fold Summary (mean ± sample standard deviation)");
const summaryHeader = [
  "dataset_key", "data_name", "method", "fold_count",
  ...metricColumns.flatMap(([name]) => [`${name}_mean`, `${name}_std`]),
];
summary.getRange("A3:V3").values = [summaryHeader];
styleHeader(summary.getRange("A3:V3"));
const identityRows = groupRanges.map((group) => [
  group.datasetKey, group.dataName, group.method,
]);
summary.getRange(`A4:C${groupRanges.length + 3}`).values = identityRows;

for (let i = 0; i < groupRanges.length; i += 1) {
  const outRow = i + 4;
  const group = groupRanges[i];
  summary.getRange(`D${outRow}`).formulas = [[
    `=COUNT('Fold Metrics'!D${group.startRow}:D${group.endRow})`,
  ]];
  let outCol = 4;
  for (const [, sourceCol] of metricColumns) {
    const meanCell = summary.getCell(outRow - 1, outCol);
    meanCell.formulas = [[
      `=AVERAGE('Fold Metrics'!${sourceCol}${group.startRow}:${sourceCol}${group.endRow})`,
    ]];
    const stdCell = summary.getCell(outRow - 1, outCol + 1);
    stdCell.formulas = [[
      `=STDEV.S('Fold Metrics'!${sourceCol}${group.startRow}:${sourceCol}${group.endRow})`,
    ]];
    outCol += 2;
  }
}
summary.getRange(`D4:D${groupRanges.length + 3}`).format.numberFormat = "0";
summary.getRange(`E4:V${groupRanges.length + 3}`).format.numberFormat = "0.0000";
summary.getRange("A:A").format.columnWidth = 22;
summary.getRange("B:B").format.columnWidth = 44;
summary.getRange("C:C").format.columnWidth = 24;
summary.getRange("D:D").format.columnWidth = 12;
summary.getRange("E:V").format.columnWidth = 15;
summary.getRange(`E4:E${groupRanges.length + 3}`).conditionalFormats.add("colorScale", {
  colors: ["#FEE2E2", "#FEF3C7", "#DCFCE7"],
  thresholds: ["min", "50%", "max"],
});
summary.freezePanes.freezeRows(3);
summary.freezePanes.freezeColumns(3);
summary.showGridLines = false;
const summaryTable = summary.tables.add(
  `A3:V${groupRanges.length + 3}`, true, "SummaryMetricsTable",
);
summaryTable.showBandedColumns = false;

// Configurations / adaptations
styleTitle(configs, "A1:H1", "Model configurations and implementation notes");
const configRows = [
  ["Method", "Input", "Core model", "Loss", "Optimizer", "Training", "Selection", "Adaptation / deviation"],
  ["GAT+mean pool", "MISGL subgraphs", "GAT encoder + mean pool", "Project formal config", "Project formal config", "Fixed protocol", "Validation only", "Existing MISGL Stage-1 branch reused."],
  ["MIL-HEAD", "MISGL subgraphs", "GAT + instance attention", "Project formal config", "Project formal config", "Fixed protocol", "Validation only", "Existing MISGL Stage-1 branch reused."],
  ["POS-HEAD", "Frozen z_mean + coarse graph", "GCN position head; top-k=16", "Project formal config", "Adam, lr=1e-3, wd=5e-4", "300 epochs, patience 50, dropout 0.5", "Validation loss", "No GAT retraining; frozen Stage-1 embeddings."],
  ["MISGL", "Frozen z_mil + coarse graph", "MIL-HEAD + GCN position head", "Project formal config", "Adam, lr=1e-3, wd=5e-4", "300 epochs, patience 50, dropout 0.5", "Validation loss", "No MIL-HEAD retraining; frozen Stage-1 embeddings."],
  ["Attention-based MIL", "Subgraph=bag; nodes=instances; edges ignored", "Gated embedding attention; hidden/attention=128", "BCE", "Adam, lr=1e-3, wd=1e-4", "200 epochs, patience 30, batch 128", "Validation loss", "Official default attention dimension; no graph structure."],
  ["RGMIL", "Subgraph=bag; nodes=instances; source edges ignored", "exp(-Euclidean) bag graph + VDN threshold/depth + GAT", "BCE", "Adam, lr=5e-4, wd=1e-3", "10,000 max epochs, patience 20; dropout 0.2", "VDN on fold 0; reuse action", "Restored paper similarity/logit/eval semantics; source graph ignored."],
  ["SubGNN", "Base graph + labeled node-set subgraphs", "All N/P/S internal+border channels; 1 layer; 8 anchors", "Cross entropy", "Adam, lr=1e-3", "200 epochs, patience 30, batch 32; FFN 64/32", "Validation loss", "Provided node features replace unavailable pretrained GIN; exact scalability/compatibility fixes."],
];
configs.getRange("A3:H10").values = configRows;
styleHeader(configs.getRange("A3:H3"));
configs.getRange("A4:A10").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#16324F" },
};
configs.getRange("A3:H10").format.wrapText = true;
configs.getRange("A:A").format.columnWidth = 24;
configs.getRange("B:B").format.columnWidth = 34;
configs.getRange("C:C").format.columnWidth = 42;
configs.getRange("D:E").format.columnWidth = 26;
configs.getRange("F:G").format.columnWidth = 34;
configs.getRange("H:H").format.columnWidth = 52;
configs.getRange("A4:H10").format.rowHeight = 58;
configs.freezePanes.freezeRows(3);
configs.showGridLines = false;
const configTable = configs.tables.add("A3:H10", true, "ConfigurationsTable");
configTable.showBandedColumns = false;

// Formula-backed validation sheet
styleTitle(validation, "A1:D1", "Workbook integrity checks");
validation.getRange("A3:D3").values = [["Check", "Expected", "Actual", "Status"]];
styleHeader(validation.getRange("A3:D3"));
validation.getRange("A4:B7").values = [
  ["Fold records", 1050],
  ["Dataset-method groups", 105],
  ["Minimum folds per group", 10],
  ["Maximum folds per group", 10],
];
validation.getRange("C4:C7").formulas = [
  ["=COUNTA('Fold Metrics'!A2:A1051)"],
  ["=COUNTA('Summary'!A4:A108)"],
  ["=MIN('Summary'!D4:D108)"],
  ["=MAX('Summary'!D4:D108)"],
];
validation.getRange("D4").formulas = [["=IF(B4=C4,\"PASS\",\"FAIL\")"]];
validation.getRange("D4:D7").fillDown();
validation.getRange("D4:D7").conditionalFormats.add("containsText", {
  text: "PASS", format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } },
});
validation.getRange("D4:D7").conditionalFormats.add("containsText", {
  text: "FAIL", format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } },
});
validation.getRange("A4:A7").format = {
  fill: "#D9EAF7", font: { bold: true, color: "#16324F" },
};
validation.getRange("A:A").format.columnWidth = 30;
validation.getRange("B:D").format.columnWidth = 18;
validation.getRange("B4:C7").format.numberFormat = "0";
validation.showGridLines = false;
validation.freezePanes.freezeRows(3);

await fs.mkdir(path.dirname(outputXlsx), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const keySummary = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:V12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 22,
});
console.log(keySummary.ndjson);
const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

for (const [sheetName, range] of [
  ["README", "A1:B17"],
  ["Summary", "A1:V20"],
  ["Fold Metrics", "A1:U25"],
  ["Configurations", "A1:H10"],
  ["Validation", "A1:D7"],
]) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputXlsx);
console.log(JSON.stringify({
  outputXlsx,
  foldRecords: foldRows.length - 1,
  groups: groupRanges.length,
}));
