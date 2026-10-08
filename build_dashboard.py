import pandas as pd
import json

df_def = pd.read_csv(r"C:\Users\Bhushan\L2D with semantic drift analysis\phase6_deferred_queue.csv")
df_ds = pd.read_csv(r"C:\Users\Bhushan\L2D with semantic drift analysis\phase6_drift_summary.csv")
df_ft = pd.read_csv(r"C:\Users\Bhushan\L2D with semantic drift analysis\phase6_feature_drift_trend.csv")

# 1. Scatter points
scatter_pts = []
for _, row in df_def.iterrows():
    scatter_pts.append({
        "id": int(row["customer_index"]),
        "w": int(row["window_id"]),
        "u": round(float(row["uncertainty_score_U"]), 4),
        "d": round(float(row["drift_severity_D"]), 4),
        "score": round(float(row["composite_score"]), 4),
        "err": int(row["is_model_error"]),
        "actual": int(row["actual_churn"]),
        "pred": int(row["model_prediction"])
    })

# 2. Table rows (top 20)
top_table = []
for _, row in df_def.head(20).iterrows():
    top_table.append({
        "id": int(row["customer_index"]),
        "w": f"W{int(row['window_id'])}",
        "score": round(float(row["composite_score"]), 3),
        "u": round(float(row["uncertainty_score_U"]), 3),
        "d": round(float(row["drift_severity_D"]), 3),
        "prob": f"{round(float(row['predicted_churn_prob'])*100, 1)}%",
        "tenure": f"{int(row['tenure'])} mo",
        "monthly": f"${float(row['MonthlyCharges']):.2f}",
        "exp": "; ".join(filter(None, [str(row['shap_explanation_1']), str(row['shap_explanation_2'])]))
    })

# 3. Heatmap (top 10 drifting features)
top10_ft = df_ft.sort_values(by="9", ascending=False).head(10)
heatmap_data = []
for _, row in top10_ft.iterrows():
    heatmap_data.append({
        "feature": str(row["feature"]),
        "effects": [round(float(row[str(w)]), 3) for w in range(1, 10)]
    })

# 4. Drift summary table
drift_summary_rows = []
for _, row in df_ds.iterrows():
    val = float(row["max_effect_size"])
    level = "HIGH" if val >= 0.91 else ("MEDIUM" if val >= 0.8 else "LOW")
    drift_summary_rows.append({
        "window": f"W{int(row['window_id'])}",
        "n_drifted": int(row['n_features_drifted']),
        "pct_drifted": f"{float(row['pct_features_drifted']):.1f}%",
        "avg_effect": f"{float(row['avg_effect_size']):.3f}",
        "top_feature": str(row['max_effect_feature']),
        "max_effect": f"{val:.3f}",
        "retrain_level": level
    })

# Compute exact KPIs
total_deferred = len(df_def)
avg_score = round(float(df_def["composite_score"].mean()), 3)
err_rate = round(float(df_def["is_model_error"].mean()) * 100, 1)
avg_churn_prob = round(float(df_def["predicted_churn_prob"].mean()) * 100, 1)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>L2D Churn Deferral — Interactive Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0f1117;
            --bg-secondary: #1a1d27;
            --bg-card: #222639;
            --bg-card-hover: #2a2f45;
            --text-primary: #e8eaed;
            --text-secondary: #9aa0b4;
            --text-muted: #6b7185;
            --accent-blue: #4f8cff;
            --accent-purple: #8b5cf6;
            --accent-teal: #14b8a6;
            --accent-orange: #f59e0b;
            --accent-red: #ef4444;
            --accent-green: #22c55e;
            --border: #2d3250;
            --glow-blue: rgba(79, 140, 255, 0.15);
            --glow-purple: rgba(139, 92, 246, 0.15);
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }}

        .dashboard-container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }}

        /* Tab Navigation */
        .tab-nav {{
            display: flex;
            gap: 4px;
            margin-bottom: 24px;
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border);
        }}

        .tab-btn {{
            flex: 1;
            padding: 12px 24px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 14px;
            font-weight: 500;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }}

        .tab-btn.active {{
            background: var(--accent-blue);
            color: white;
            box-shadow: 0 2px 8px rgba(79, 140, 255, 0.3);
        }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .dashboard-header {{
            margin-bottom: 24px;
        }}

        .dashboard-header h1 {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 6px;
        }}

        .dashboard-header p {{
            color: var(--text-secondary);
            font-size: 13px;
        }}

        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }}

        .kpi-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 4px;
        }}

        .kpi-sub {{
            font-size: 11px;
            color: var(--text-secondary);
        }}

        /* Filters */
        .filter-bar {{
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 14px;
        }}

        .filter-group label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
            white-space: nowrap;
        }}

        .filter-group select,
        .filter-group input {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-family: 'Inter', sans-serif;
        }}

        /* Charts Grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }}

        .chart-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            position: relative;
        }}

        .chart-card.full-width {{
            grid-column: 1 / -1;
        }}

        .chart-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .chart-counter {{
            font-size: 11px;
            font-weight: 400;
            color: var(--text-secondary);
        }}

        /* Real Scatter Plot Canvas / Container */
        .scatter-container {{
            height: 280px;
            border-radius: 8px;
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            position: relative;
            overflow: hidden;
        }}

        .scatter-axis-label-x {{
            position: absolute;
            bottom: 4px;
            right: 12px;
            font-size: 10px;
            color: var(--text-muted);
        }}

        .scatter-axis-label-y {{
            position: absolute;
            top: 8px;
            left: 8px;
            font-size: 10px;
            color: var(--text-muted);
        }}

        .scatter-dot {{
            position: absolute;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            transition: transform 0.15s ease, opacity 0.15s ease;
            cursor: pointer;
        }}

        .scatter-dot:hover {{
            transform: scale(2.2);
            z-index: 100;
            box-shadow: 0 0 8px rgba(255, 255, 255, 0.8);
        }}

        .scatter-dot.error {{
            background: var(--accent-red);
            opacity: 0.8;
        }}

        .scatter-dot.correct {{
            background: var(--accent-blue);
            opacity: 0.45;
        }}

        /* Tooltip */
        #tooltip {{
            position: absolute;
            display: none;
            background: #111420;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 11px;
            color: var(--text-primary);
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            line-height: 1.4;
        }}

        /* Bar chart */
        .bar-chart {{
            display: flex;
            align-items: flex-end;
            gap: 12px;
            height: 240px;
            padding: 16px 8px;
        }}

        .bar-group {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }}

        .bar {{
            width: 100%;
            max-width: 44px;
            border-radius: 4px 4px 0 0;
            transition: height 0.3s ease;
        }}

        .bar-label {{
            font-size: 10px;
            color: var(--text-muted);
        }}

        .bar-value {{
            font-size: 10px;
            font-weight: 600;
            color: var(--text-secondary);
        }}

        /* Table */
        .data-table-container {{
            overflow-x: auto;
            max-height: 420px;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}

        .data-table th {{
            background: var(--bg-secondary);
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.5px;
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 5;
        }}

        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }}

        .data-table tr:hover td {{
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }}

        /* Badges */
        .score-badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}

        .score-high {{ background: rgba(239, 68, 68, 0.18); color: var(--accent-red); }}
        .score-medium {{ background: rgba(245, 158, 11, 0.18); color: var(--accent-orange); }}
        .score-low {{ background: rgba(34, 197, 94, 0.18); color: var(--accent-green); }}

        /* Heatmap */
        .heatmap {{
            display: grid;
            gap: 3px;
        }}

        .heatmap-row {{
            display: flex;
            gap: 3px;
            align-items: center;
        }}

        .heatmap-label {{
            width: 220px;
            font-size: 11px;
            color: var(--text-secondary);
            text-align: right;
            padding-right: 12px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .heatmap-cell {{
            flex: 1;
            height: 28px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 500;
            min-width: 48px;
        }}

        .heatmap-header {{
            display: flex;
            gap: 3px;
            margin-left: 220px;
            margin-bottom: 6px;
        }}

        .heatmap-header span {{
            flex: 1;
            text-align: center;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            min-width: 48px;
        }}

        /* Status indicators */
        .status-pill {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}

        .status-high {{ background: rgba(239, 68, 68, 0.2); color: var(--accent-red); }}
        .status-medium {{ background: rgba(245, 158, 11, 0.2); color: var(--accent-orange); }}
        .status-low {{ background: rgba(34, 197, 94, 0.2); color: var(--accent-green); }}

        /* Legend */
        .legend {{
            display: flex;
            gap: 16px;
            margin-top: 10px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            color: var(--text-secondary);
        }}

        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}

        .verified-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.25);
            color: var(--accent-green);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            margin-bottom: 12px;
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="verified-badge">
            ✓ Real Verified Data Ingested — Global D(x) Normalization Fix Applied across Windows 1–9
        </div>

        <!-- Tab Navigation -->
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('analyst')">
                👤 Analyst Deferral Queue (Real Pipeline Output)
            </button>
            <button class="tab-btn" onclick="switchTab('mlteam')">
                🔬 ML Team Drift Monitor (Phase 3 &amp; 6 Drift Metrics)
            </button>
        </div>

        <!-- ═══════════════════════════════════════════════ -->
        <!-- TAB 1: ANALYST DEFERRAL QUEUE                   -->
        <!-- ═══════════════════════════════════════════════ -->
        <div id="tab-analyst" class="tab-content active">
            <div class="dashboard-header">
                <h1>Churn Deferral Queue</h1>
                <p>Real deferred cases from the calibrated pipeline — ranked by composite score Score(x) = U(x) + D(x) + 0.15·U·D with global D(x) normalization</p>
            </div>

            <!-- KPI Cards -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Cases in Queue</div>
                    <div class="kpi-value" id="kpi-cases" style="color: var(--accent-blue);">{total_deferred}</div>
                    <div class="kpi-sub">15% budget (105 / window across W1-W9)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg Composite Score</div>
                    <div class="kpi-value" id="kpi-score" style="color: var(--accent-purple);">{avg_score}</div>
                    <div class="kpi-sub">Score(x) with global D(x) normalization</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Model Errors Caught</div>
                    <div class="kpi-value" id="kpi-precision" style="color: var(--accent-orange);">{err_rate}%</div>
                    <div class="kpi-sub">449 / 945 error cases caught (41.9% on W3-W9)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Avg Calibrated Churn Prob</div>
                    <div class="kpi-value" id="kpi-churn" style="color: var(--accent-teal);">{avg_churn_prob}%</div>
                    <div class="kpi-sub">Mean U(x) among deferred customers</div>
                </div>
            </div>

            <!-- Filters -->
            <div class="filter-bar">
                <div class="filter-group">
                    <label>Window</label>
                    <select id="window-filter" onchange="filterData()">
                        <option value="all">All Windows (1-9)</option>
                        <option value="1">Window 1</option>
                        <option value="2">Window 2</option>
                        <option value="3">Window 3</option>
                        <option value="4">Window 4</option>
                        <option value="5">Window 5</option>
                        <option value="6">Window 6</option>
                        <option value="7">Window 7</option>
                        <option value="8">Window 8</option>
                        <option value="9">Window 9</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Outcome Filter</label>
                    <select id="outcome-filter" onchange="filterData()">
                        <option value="all">All Cases</option>
                        <option value="error">Model Errors Only</option>
                        <option value="correct">Correct Predictions Only</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Min Score</label>
                    <input type="range" id="score-slider" min="0" max="150" value="0" style="width: 90px;" oninput="updateScoreFilter(this.value)">
                    <span id="score-val" style="font-size: 11px; color: var(--text-secondary);">≥ 0.0</span>
                </div>
            </div>

            <!-- Charts Row -->
            <div class="charts-grid">
                <div class="chart-card">
                    <div class="chart-title">
                        <span>U(x) vs D(x) — Real Deferred Cases</span>
                        <span class="chart-counter" id="scatter-counter">Showing 945 points</span>
                    </div>
                    <div class="scatter-container" id="scatter-chart">
                        <div class="scatter-axis-label-x">Calibrated Uncertainty U(x) →</div>
                        <div class="scatter-axis-label-y">↑ Global Drift Severity D(x)</div>
                        <!-- Real dots rendered by JS -->
                    </div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-dot" style="background: var(--accent-red);"></div> Model Error (Deferred &amp; Caught)</div>
                        <div class="legend-item"><div class="legend-dot" style="background: var(--accent-blue);"></div> Correct Model Prediction</div>
                    </div>
                </div>
                <div class="chart-card">
                    <div class="chart-title">Deferred Cases by Temporal Window</div>
                    <div class="bar-chart" id="bar-chart-container">
                        <!-- Bars dynamically rendered -->
                    </div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-dot" style="background: var(--accent-purple);"></div> 105 deferred cases per window (exact 15% budget cap)</div>
                    </div>
                </div>
            </div>

            <!-- Deferral Queue Table -->
            <div class="chart-card full-width">
                <div class="chart-title">
                    <span>Deferral Queue — Top Cases by Composite Score</span>
                    <span class="chart-counter">Top 20 from phase6_deferred_queue.csv</span>
                </div>
                <div class="data-table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Customer ID</th>
                                <th>Window</th>
                                <th>Score(x)</th>
                                <th>U(x)</th>
                                <th>D(x)</th>
                                <th>Churn Prob</th>
                                <th>Tenure</th>
                                <th>Monthly $</th>
                                <th>SHAP Attribution Highlights</th>
                            </tr>
                        </thead>
                        <tbody id="queue-table-body">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- ═══════════════════════════════════════════════ -->
        <!-- TAB 2: ML TEAM DRIFT MONITOR                    -->
        <!-- ═══════════════════════════════════════════════ -->
        <div id="tab-mlteam" class="tab-content">
            <div class="dashboard-header">
                <h1>ML Team Drift Monitor</h1>
                <p>Explanation drift tracking via Mann-Whitney U test on SHAP distributions relative to baseline Window 0</p>
            </div>

            <!-- ML Team KPIs -->
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Features Tracked</div>
                    <div class="kpi-value" style="color: var(--accent-blue);">32</div>
                    <div class="kpi-sub">Total model input features</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Features Drifted (W9)</div>
                    <div class="kpi-value" style="color: var(--accent-red);">20 / 32</div>
                    <div class="kpi-sub">62.5% significant drift (p &lt; 0.05)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Max Drift Feature</div>
                    <div class="kpi-value" style="color: var(--accent-orange); font-size: 20px; line-height: 38px;">Contract_Two yr</div>
                    <div class="kpi-sub">Effect size = 0.926 in Window 9</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Retrain Alert Status</div>
                    <div class="kpi-value" style="color: var(--accent-red);">HIGH</div>
                    <div class="kpi-sub">Exceeded threshold (effect &gt; 0.90) in W6-W9</div>
                </div>
            </div>

            <!-- Heatmap Card -->
            <div class="chart-card full-width" style="margin-bottom: 24px;">
                <div class="chart-title">Top 10 Drifting Features — Effect Size Across Windows (Phase 3 &amp; 6)</div>
                <div class="heatmap-header">
                    <span>W1</span><span>W2</span><span>W3</span><span>W4</span><span>W5</span>
                    <span>W6</span><span>W7</span><span>W8</span><span>W9</span>
                </div>
                <div class="heatmap" id="heatmap">
                    <!-- Populated dynamically -->
                </div>
                <div class="legend" style="margin-top: 14px;">
                    <div class="legend-item"><div class="legend-dot" style="background: rgba(79, 140, 255, 0.2);"></div> &lt; 0.20 (Mild)</div>
                    <div class="legend-item"><div class="legend-dot" style="background: rgba(79, 140, 255, 0.55);"></div> 0.20 – 0.60 (Moderate)</div>
                    <div class="legend-item"><div class="legend-dot" style="background: rgba(79, 140, 255, 0.95);"></div> &gt; 0.60 (Severe Shift)</div>
                </div>
            </div>

            <!-- Drift Summary Table -->
            <div class="chart-card full-width">
                <div class="chart-title">Drift Summary by Temporal Window</div>
                <div class="data-table-container">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Window</th>
                                <th>Features Drifted</th>
                                <th>% Drifted</th>
                                <th>Avg Effect Size</th>
                                <th>Top Drifting Feature</th>
                                <th>Max Effect Size</th>
                                <th>Retrain Signal</th>
                            </tr>
                        </thead>
                        <tbody id="drift-summary-body">
                            <!-- Populated dynamically -->
                        </tbody>
                    </table>
                </div>
                <div style="margin-top: 12px; font-size: 11px; color: var(--text-muted);">
                    Data source: <code>phase6_drift_summary.csv</code> and <code>phase6_feature_drift_trend.csv</code>
                </div>
            </div>
        </div>
    </div>

    <!-- Floating Tooltip -->
    <div id="tooltip"></div>

    <script>
        // Real Data Arrays
        const realScatterData = {json.dumps(scatter_pts)};
        const realTopTable = {json.dumps(top_table)};
        const realHeatmapData = {json.dumps(heatmap_data)};
        const realDriftSummary = {json.dumps(drift_summary_rows)};

        // Tab switching
        function switchTab(tab) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tab).classList.add('active');
            event.target.classList.add('active');
        }}

        // Render Scatter Plot
        const scatterChart = document.getElementById('scatter-chart');
        const tooltip = document.getElementById('tooltip');
        let minScoreFilter = 0;

        function renderScatter(pts) {{
            // Remove previous dots
            scatterChart.querySelectorAll('.scatter-dot').forEach(el => el.remove());

            // Normalization ranges for coordinates: U in [0, 0.40], D in [0, 1.0]
            const maxU = 0.40;
            const maxD = 1.02;

            pts.forEach(p => {{
                const dot = document.createElement('div');
                dot.className = 'scatter-dot ' + (p.err === 1 ? 'error' : 'correct');
                
                const leftPct = Math.min(95, Math.max(5, (p.u / maxU) * 88 + 5));
                const bottomPct = Math.min(95, Math.max(5, (p.d / maxD) * 85 + 6));
                dot.style.left = leftPct + '%';
                dot.style.bottom = bottomPct + '%';

                dot.addEventListener('mouseenter', (e) => {{
                    tooltip.style.display = 'block';
                    tooltip.innerHTML = `
                        <strong>Customer #${{p.id}}</strong> (Window ${{p.w}})<br>
                        Composite Score: <strong>${{p.score}}</strong><br>
                        U(x) Uncertainty: ${{p.u}}<br>
                        D(x) Global Drift: ${{p.d}}<br>
                        Model Prediction: ${{p.pred}} | Actual: ${{p.actual}}<br>
                        Status: <span style="color: ${{p.err === 1 ? '#ef4444' : '#4f8cff'}};">${{p.err === 1 ? 'Model Error Caught' : 'Correct Prediction'}}</span>
                    `;
                    positionTooltip(e);
                }});

                dot.addEventListener('mousemove', positionTooltip);
                dot.addEventListener('mouseleave', () => {{
                    tooltip.style.display = 'none';
                }});

                scatterChart.appendChild(dot);
            }});

            document.getElementById('scatter-counter').textContent = `Showing ${{pts.length}} real deferred points`;
        }}

        function positionTooltip(e) {{
            tooltip.style.left = (e.pageX + 12) + 'px';
            tooltip.style.top = (e.pageY + 12) + 'px';
        }}

        // Filters
        function updateScoreFilter(val) {{
            minScoreFilter = (val / 100);
            document.getElementById('score-val').textContent = '≥ ' + minScoreFilter.toFixed(2);
            filterData();
        }}

        function filterData() {{
            const win = document.getElementById('window-filter').value;
            const outcome = document.getElementById('outcome-filter').value;

            let filtered = realScatterData.filter(p => {{
                if (win !== 'all' && p.w !== parseInt(win)) return false;
                if (outcome === 'error' && p.err !== 1) return false;
                if (outcome === 'correct' && p.err !== 0) return false;
                if (p.score < minScoreFilter) return false;
                return true;
            }});

            renderScatter(filtered);

            // Update Dynamic KPIs based on filtered
            if (filtered.length > 0) {{
                document.getElementById('kpi-cases').textContent = filtered.length;
                const avgSc = filtered.reduce((acc, p) => acc + p.score, 0) / filtered.length;
                document.getElementById('kpi-score').textContent = avgSc.toFixed(3);
                const errors = filtered.filter(p => p.err === 1).length;
                const prec = (errors / filtered.length * 100).toFixed(1);
                document.getElementById('kpi-precision').textContent = prec + '%';
                const avgU = filtered.reduce((acc, p) => acc + p.u, 0) / filtered.length;
                document.getElementById('kpi-churn').textContent = (avgU * 100).toFixed(1) + '%';
            }}
        }}

        // Render Bar Chart
        function renderBarChart() {{
            const container = document.getElementById('bar-chart-container');
            container.innerHTML = '';
            for (let w = 1; w <= 9; w++) {{
                const group = document.createElement('div');
                group.className = 'bar-group';
                group.innerHTML = `
                    <div class="bar-value">105</div>
                    <div class="bar" style="height: 100%; background: linear-gradient(180deg, var(--accent-blue), var(--accent-purple));"></div>
                    <div class="bar-label">W${{w}}</div>
                `;
                container.appendChild(group);
            }}
        }}

        // Render Table
        function renderTable() {{
            const tbody = document.getElementById('queue-table-body');
            tbody.innerHTML = '';
            realTopTable.forEach(row => {{
                const tr = document.createElement('tr');
                const badgeClass = row.score >= 1.2 ? 'score-high' : (row.score >= 0.8 ? 'score-medium' : 'score-low');
                tr.innerHTML = `
                    <td><strong>#${{row.id}}</strong></td>
                    <td>${{row.w}}</td>
                    <td><span class="score-badge ${{badgeClass}}">${{row.score}}</span></td>
                    <td>${{row.u}}</td>
                    <td>${{row.d}}</td>
                    <td>${{row.prob}}</td>
                    <td>${{row.tenure}}</td>
                    <td>${{row.monthly}}</td>
                    <td style="font-size: 11px; color: var(--text-muted);">${{row.exp}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        // Render Heatmap
        function renderHeatmap() {{
            const container = document.getElementById('heatmap');
            container.innerHTML = '';
            realHeatmapData.forEach(item => {{
                const row = document.createElement('div');
                row.className = 'heatmap-row';

                const label = document.createElement('div');
                label.className = 'heatmap-label';
                label.textContent = item.feature;
                label.title = item.feature;
                row.appendChild(label);

                item.effects.forEach(effect => {{
                    const cell = document.createElement('div');
                    cell.className = 'heatmap-cell';
                    // Alpha based on magnitude
                    const alpha = Math.min(0.95, Math.max(0.12, effect));
                    cell.style.background = `rgba(79, 140, 255, ${{alpha}})`;
                    cell.textContent = effect.toFixed(2);
                    cell.style.color = alpha > 0.45 ? '#ffffff' : 'rgba(255, 255, 255, 0.6)';
                    row.appendChild(cell);
                }});

                container.appendChild(row);
            }});
        }}

        // Render Drift Summary
        function renderDriftSummary() {{
            const tbody = document.getElementById('drift-summary-body');
            tbody.innerHTML = '';
            realDriftSummary.forEach(row => {{
                const tr = document.createElement('tr');
                const pillClass = row.retrain_level === 'HIGH' ? 'status-high' : (row.retrain_level === 'MEDIUM' ? 'status-medium' : 'status-low');
                tr.innerHTML = `
                    <td><strong>${{row.window}}</strong></td>
                    <td>${{row.n_drifted}} / 32</td>
                    <td>${{row.pct_drifted}}</td>
                    <td>${{row.avg_effect}}</td>
                    <td><code>${{row.top_feature}}</code></td>
                    <td><strong>${{row.max_effect}}</strong></td>
                    <td><span class="status-pill ${{pillClass}}">${{row.retrain_level}}</span></td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        // Initial execution
        renderScatter(realScatterData);
        renderBarChart();
        renderTable();
        renderHeatmap();
        renderDriftSummary();
    </script>
</body>
</html>
"""

with open(r"c:\L2D with semantic drift analysis\Phase6_dashboard_sketch.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Phase6_dashboard_sketch.html successfully updated with real computed data.")
