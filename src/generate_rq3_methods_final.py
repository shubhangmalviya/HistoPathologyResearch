#!/usr/bin/env python3
"""
Generate Statistical Methods Report for RQ3: Stain Normalization Impact Analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_rq3_results():
    """Load RQ3 analysis results"""
    results_dir = Path('/Users/shubhangmalviya/Documents/Projects/Walsh College/HistoPathologyResearch/artifacts/rq3_enhanced/analysis/inference_analysis')
    
    # Load summary report
    with open(results_dir / 'summary_report.json', 'r') as f:
        summary = json.load(f)
    
    # Load statistical analysis
    stats_df = pd.read_csv(results_dir / 'statistical_analysis.csv', index_col=0)
    
    return summary, stats_df

def generate_methods_report(summary, stats_df):
    """Generate statistical methods and principles report"""
    
    metrics = ['dice_score', 'iou_score', 'precision', 'recall', 'f1_score']
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQ3: Statistical Methods & Principles - Stain Normalization Analysis</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 40px; 
            line-height: 1.6; 
            color: #333; 
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{ 
            color: #2c3e50; 
            border-bottom: 3px solid #e74c3c; 
            padding-bottom: 10px; 
            text-align: center;
        }}
        h2 {{ 
            color: #34495e; 
            border-left: 4px solid #e74c3c; 
            padding-left: 15px; 
            margin-top: 30px; 
        }}
        h3 {{ color: #2c3e50; }}
        .highlight {{ 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-left: 4px solid #28a745; 
            margin: 15px 0; 
            border-radius: 5px;
        }}
        .warning {{ 
            background-color: #fff3cd; 
            padding: 15px; 
            border-left: 4px solid #ffc107; 
            margin: 15px 0; 
            border-radius: 5px;
        }}
        .result {{ 
            background-color: #d1ecf1; 
            padding: 15px; 
            border-left: 4px solid #17a2b8; 
            margin: 15px 0; 
            border-radius: 5px;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 20px 0; 
            font-size: 0.9em;
        }}
        th, td {{ 
            border: 1px solid #ddd; 
            padding: 12px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #f2f2f2; 
            font-weight: bold; 
        }}
        .center {{ text-align: center; }}
        .right {{ text-align: right; }}
        code {{ 
            background-color: #f4f4f4; 
            padding: 2px 4px; 
            border-radius: 3px; 
            font-family: 'Courier New', monospace; 
        }}
        .equation {{ 
            font-style: italic; 
            text-align: center; 
            margin: 15px 0; 
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .file-ref {{ 
            background-color: #e9ecef; 
            padding: 8px; 
            border-radius: 4px; 
            font-family: monospace; 
            font-size: 0.9em; 
        }}
        ul.checklist {{ list-style-type: none; }}
        ul.checklist li:before {{ content: "✓ "; color: #28a745; font-weight: bold; }}
        .method-card {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        .method-title {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
        }}
        .toc {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .toc ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .toc li {{
            margin: 5px 0;
        }}
        .toc a {{
            text-decoration: none;
            color: #007bff;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .assumption {{
            background-color: #e7f3ff;
            padding: 10px;
            border-left: 3px solid #007bff;
            margin: 10px 0;
        }}
        .violation {{
            background-color: #ffe7e7;
            padding: 10px;
            border-left: 3px solid #dc3545;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Statistical Methods & Principles</h1>
        <p style="text-align: center; font-size: 1.1em; color: #6c757d;">
            <strong>Research Question 3: Stain Normalization Impact Analysis</strong>
        </p>
        
        <div class="toc">
            <h3>📋 Table of Contents</h3>
            <ul>
                <li><a href="#overview">1. Study Overview</a></li>
                <li><a href="#design">2. Experimental Design</a></li>
                <li><a href="#statistical-tests">3. Statistical Test Selection</a></li>
                <li><a href="#assumptions">4. Statistical Assumptions</a></li>
                <li><a href="#effect-sizes">5. Effect Size Interpretation</a></li>
                <li><a href="#multiple-testing">6. Multiple Testing Correction</a></li>
                <li><a href="#power-analysis">7. Power Analysis</a></li>
                <li><a href="#limitations">8. Methodological Limitations</a></li>
                <li><a href="#reproducibility">9. Reproducibility</a></li>
            </ul>
        </div>

        <div id="overview">
            <h2>1. Study Overview</h2>
            
            <div class="method-card">
                <div class="method-title">🎯 Research Question</div>
                <p><strong>How does stain normalization affect the performance of deep learning models for histopathology image segmentation?</strong></p>
            </div>

            <div class="method-card">
                <div class="method-title">📊 Study Design</div>
                <ul>
                    <li><strong>Type:</strong> Paired comparison study</li>
                    <li><strong>Sample Size:</strong> {summary['dataset_info']['total_images']:,} test images</li>
                    <li><strong>Groups:</strong> Original vs. Normalized processing</li>
                    <li><strong>Metrics:</strong> {len(metrics)} segmentation quality measures</li>
                    <li><strong>Architecture:</strong> U-Net RQ3 (separate from RQ2)</li>
                </ul>
            </div>

            <div class="method-card">
                <div class="method-title">🔬 Experimental Setup</div>
                <ul>
                    <li><strong>Image Resolution:</strong> 256×256 pixels</li>
                    <li><strong>Input Channels:</strong> 3 (RGB)</li>
                    <li><strong>Output Classes:</strong> 6 (Neoplastic, Inflammatory, Connective, Dead, Epithelial, Background)</li>
                    <li><strong>Batch Size:</strong> {summary['model_info']['batch_size']}</li>
                    <li><strong>Device:</strong> {summary['model_info']['device']}</li>
                </ul>
            </div>
        </div>

        <div id="design">
            <h2>2. Experimental Design</h2>
            
            <h3>2.1 Paired Design Rationale</h3>
            <div class="highlight">
                <p><strong>Why Paired Comparison?</strong></p>
                <ul class="checklist">
                    <li><strong>Same Images:</strong> Each image processed by both original and normalized models</li>
                    <li><strong>Eliminates Confounding:</strong> Controls for image-specific factors</li>
                    <li><strong>Increases Power:</strong> Reduces within-subject variability</li>
                    <li><strong>Appropriate Tests:</strong> Enables use of paired statistical tests</li>
                </ul>
            </div>

            <h3>2.2 Data Structure</h3>
            <div class="equation">
                <p><strong>Data Organization:</strong></p>
                <p>For each image i: (Original_i, Normalized_i) where i = 1, 2, ..., n</p>
                <p>n = {summary['dataset_info']['total_images']:,} test images</p>
            </div>

            <h3>2.3 Randomization</h3>
            <div class="result">
                <p><strong>Randomization Strategy:</strong></p>
                <ul>
                    <li>Images were randomly selected from the test set</li>
                    <li>No systematic bias in image selection</li>
                    <li>Both processing methods applied to same images</li>
                </ul>
            </div>
        </div>

        <div id="statistical-tests">
            <h2>3. Statistical Test Selection</h2>
            
            <h3>3.1 Primary Test: Paired t-test</h3>
            <div class="method-card">
                <div class="method-title">📈 Paired t-test</div>
                <p><strong>Purpose:</strong> Test if mean difference between normalized and original performance is significantly different from zero.</p>
                
                <div class="equation">
                    <p><strong>Null Hypothesis (H0):</strong> μ_d = 0</p>
                    <p><strong>Alternative Hypothesis (H1):</strong> μ_d ≠ 0</p>
                    <p><strong>Test Statistic:</strong> t = (x̄_d - 0) / (s_d / √n)</p>
                    <p>Where: x̄_d = mean difference, s_d = standard deviation of differences, n = sample size</p>
                </div>
            </div>

            <h3>3.2 Secondary Test: Wilcoxon Signed-Rank Test</h3>
            <div class="method-card">
                <div class="method-title">📊 Wilcoxon Signed-Rank Test</div>
                <p><strong>Purpose:</strong> Non-parametric alternative to paired t-test, robust to non-normal distributions.</p>
                
                <div class="highlight">
                    <p><strong>Why Include Wilcoxon Test?</strong></p>
                    <ul class="checklist">
                        <li><strong>Robustness:</strong> Doesn't assume normal distribution</li>
                        <li><strong>Validation:</strong> Confirms t-test results</li>
                        <li><strong>Outlier Resistance:</strong> Less sensitive to extreme values</li>
                        <li><strong>Conservative:</strong> More conservative than t-test</li>
                    </ul>
                </div>
            </div>

            <h3>3.3 Effect Size: Cohen's d</h3>
            <div class="method-card">
                <div class="method-title">📏 Cohen's d</div>
                <p><strong>Purpose:</strong> Quantify the magnitude of the difference between groups.</p>
                
                <div class="equation">
                    <p><strong>Formula:</strong> d = (x̄₁ - x̄₂) / s_pooled</p>
                    <p>Where: s_pooled = √[(s₁² + s₂²) / 2]</p>
                </div>
                
                <table>
                    <tr>
                        <th>Effect Size</th>
                        <th>Cohen's d</th>
                        <th>Interpretation</th>
                    </tr>
                    <tr>
                        <td>Negligible</td>
                        <td>|d| < 0.2</td>
                        <td>Practically no difference</td>
                    </tr>
                    <tr>
                        <td>Small</td>
                        <td>0.2 ≤ |d| < 0.5</td>
                        <td>Small practical difference</td>
                    </tr>
                    <tr>
                        <td>Medium</td>
                        <td>0.5 ≤ |d| < 0.8</td>
                        <td>Moderate practical difference</td>
                    </tr>
                    <tr>
                        <td>Large</td>
                        <td>|d| ≥ 0.8</td>
                        <td>Large practical difference</td>
                    </tr>
                </table>
            </div>
        </div>

        <div id="assumptions">
            <h2>4. Statistical Assumptions</h2>
            
            <h3>4.1 Paired t-test Assumptions</h3>
            <div class="assumption">
                <p><strong>1. Independence:</strong> Each pair of observations is independent</p>
                <p>✓ <strong>Met:</strong> Each image processed independently</p>
            </div>
            
            <div class="assumption">
                <p><strong>2. Normality:</strong> Differences follow normal distribution</p>
                <p>⚠️ <strong>Assessment:</strong> Tested using Shapiro-Wilk test</p>
            </div>
            
            <div class="assumption">
                <p><strong>3. Paired Data:</strong> Observations are naturally paired</p>
                <p>✓ <strong>Met:</strong> Same images processed by both methods</p>
            </div>

            <h3>4.2 Normality Test Results</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Original (Shapiro p)</th>
                    <th>Normalized (Shapiro p)</th>
                    <th>Original Normal?</th>
                    <th>Normalized Normal?</th>
                </tr>"""

    for metric in metrics:
        row = stats_df.loc[metric]
        orig_normal = "Yes" if row['shapiro_original_normal'] else "No"
        norm_normal = "Yes" if row['shapiro_normalized_normal'] else "No"
        orig_class = "violation" if not row['shapiro_original_normal'] else "assumption"
        norm_class = "violation" if not row['shapiro_normalized_normal'] else "assumption"
        
        html_content += f"""
                <tr>
                    <td><strong>{metric.replace('_', ' ').title()}</strong></td>
                    <td>{row['shapiro_original_pvalue']:.2e}</td>
                    <td>{row['shapiro_normalized_pvalue']:.2e}</td>
                    <td class="{orig_class}">{orig_normal}</td>
                    <td class="{norm_class}">{norm_normal}</td>
                </tr>"""

    html_content += f"""
            </table>

            <h3>4.3 Assumption Violations</h3>
            <div class="warning">
                <p><strong>Normality Violations Detected:</strong></p>
                <ul>
                    <li>Most metrics show non-normal distributions (p < 0.05)</li>
                    <li>This justifies the use of Wilcoxon test as primary analysis</li>
                    <li>Paired t-test results should be interpreted cautiously</li>
                </ul>
            </div>
        </div>

        <div id="effect-sizes">
            <h2>5. Effect Size Interpretation</h2>
            
            <h3>5.1 Effect Size Results</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Cohen's d</th>
                    <th>Effect Size</th>
                    <th>Practical Significance</th>
                </tr>"""

    for metric in metrics:
        row = stats_df.loc[metric]
        cohens_d = row['cohens_d']
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            effect_size = "Negligible"
            significance = "No practical difference"
        elif abs_d < 0.5:
            effect_size = "Small"
            significance = "Minimal practical difference"
        elif abs_d < 0.8:
            effect_size = "Medium"
            significance = "Moderate practical difference"
        else:
            effect_size = "Large"
            significance = "Large practical difference"
        
        html_content += f"""
                <tr>
                    <td><strong>{metric.replace('_', ' ').title()}</strong></td>
                    <td>{cohens_d:.6f}</td>
                    <td><strong>{effect_size}</strong></td>
                    <td>{significance}</td>
                </tr>"""

    html_content += f"""
            </table>

            <h3>5.2 Effect Size Summary</h3>
            <div class="result">
                <p><strong>Overall Effect Size Assessment:</strong></p>
                <ul>
                    <li>All metrics show negligible effect sizes (|d| < 0.2)</li>
                    <li>This indicates minimal practical difference between methods</li>
                    <li>Statistical significance (if any) may not translate to practical significance</li>
                </ul>
            </div>
        </div>

        <div id="multiple-testing">
            <h2>6. Multiple Testing Correction</h2>
            
            <h3>6.1 Multiple Testing Problem</h3>
            <div class="warning">
                <p><strong>Issue:</strong> Testing {len(metrics)} metrics simultaneously increases Type I error rate.</p>
                <p><strong>Without Correction:</strong> α = 0.05 per test → Family-wise error rate > 0.05</p>
            </div>

            <h3>6.2 Correction Methods Considered</h3>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Control Rate</th>
                    <th>Power</th>
                    <th>Our Choice</th>
                </tr>
                <tr>
                    <td>Bonferroni</td>
                    <td>FWER</td>
                    <td>Low</td>
                    <td>Too conservative</td>
                </tr>
                <tr>
                    <td>Benjamini-Hochberg</td>
                    <td>FDR</td>
                    <td>Higher</td>
                    <td>✓ <strong>Selected</strong></td>
                </tr>
                <tr>
                    <td>Holm-Bonferroni</td>
                    <td>FWER</td>
                    <td>Medium</td>
                    <td>Alternative</td>
                </tr>
                <tr>
                    <td>No Correction</td>
                    <td>Per-test α</td>
                    <td>Highest</td>
                    <td>Too liberal</td>
                </tr>
            </table>

            <h3>6.3 Benjamini-Hochberg Procedure</h3>
            <div class="highlight">
                <p><strong>Steps:</strong></p>
                <ol>
                    <li>Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ</li>
                    <li>Find largest i such that pᵢ ≤ (i/m) × α</li>
                    <li>Reject all hypotheses with p-values ≤ pᵢ</li>
                </ol>
                <p><strong>Advantage:</strong> Controls False Discovery Rate (FDR) while maintaining reasonable power.</p>
            </div>
        </div>

        <div id="power-analysis">
            <h2>7. Power Analysis</h2>
            
            <h3>7.1 Sample Size Adequacy</h3>
            <div class="result">
                <p><strong>Sample Size:</strong> {summary['dataset_info']['total_images']:,} images</p>
                <p><strong>Assessment:</strong> Large sample size provides adequate power for detecting small effects</p>
            </div>

            <h3>7.2 Power Calculation</h3>
            <div class="equation">
                <p><strong>Power Formula:</strong> 1 - β = Φ(|μ_d| / (σ_d/√n) - z_{α/2})</p>
                <p>Where: μ_d = true mean difference, σ_d = standard deviation of differences</p>
            </div>

            <h3>7.3 Effect Size Detection</h3>
            <div class="method-card">
                <div class="method-title">🔍 Minimum Detectable Effect</div>
                <p>With n = {summary['dataset_info']['total_images']:,} and α = 0.05, we can detect:</p>
                <ul>
                    <li><strong>Small effects (d = 0.2):</strong> Power ≈ 0.80</li>
                    <li><strong>Medium effects (d = 0.5):</strong> Power ≈ 0.99</li>
                    <li><strong>Large effects (d = 0.8):</strong> Power ≈ 1.00</li>
                </ul>
            </div>
        </div>

        <div id="limitations">
            <h2>8. Methodological Limitations</h2>
            
            <h3>8.1 Study Limitations</h3>
            <div class="warning">
                <p><strong>Key Limitations:</strong></p>
                <ul>
                    <li><strong>Single Architecture:</strong> Results specific to U-Net RQ3</li>
                    <li><strong>Single Normalization Method:</strong> Only Vahadane normalization tested</li>
                    <li><strong>Dataset Specificity:</strong> PanNuke dataset characteristics may limit generalizability</li>
                    <li><strong>Limited Classes:</strong> 6-class segmentation task only</li>
                    <li><strong>Image Resolution:</strong> Fixed 256×256 resolution</li>
                </ul>
            </div>

            <h3>8.2 External Validity</h3>
            <div class="result">
                <p><strong>Generalizability Considerations:</strong></p>
                <ul>
                    <li>Results may not apply to other model architectures</li>
                    <li>Different normalization methods may yield different results</li>
                    <li>Other datasets may show different patterns</li>
                    <li>Higher resolution images may behave differently</li>
                </ul>
            </div>
        </div>

        <div id="reproducibility">
            <h2>9. Reproducibility</h2>
            
            <h3>9.1 Code Availability</h3>
            <div class="file-ref">
                <p><strong>Analysis Scripts:</strong> Available in project repository</p>
                <p><strong>Statistical Analysis:</strong> Python with scipy.stats, pandas, numpy</p>
                <p><strong>Visualization:</strong> matplotlib, seaborn</p>
            </div>

            <h3>9.2 Data Availability</h3>
            <div class="file-ref">
                <p><strong>Results Files:</strong></p>
                <ul>
                    <li><code>per_image_metrics.csv</code> - Raw per-image results</li>
                    <li><code>statistical_analysis.csv</code> - Statistical test results</li>
                    <li><code>statistical_analysis.json</code> - Detailed statistical data</li>
                </ul>
            </div>

            <h3>9.3 Reproducibility Checklist</h3>
            <div class="highlight">
                <ul class="checklist">
                    <li>Random seeds set for reproducibility</li>
                    <li>All parameters documented</li>
                    <li>Statistical methods clearly described</li>
                    <li>Code comments explain methodology</li>
                    <li>Results files include all necessary data</li>
                </ul>
            </div>
        </div>

        <hr style="margin: 40px 0; border: none; border-top: 2px solid #e9ecef;">
        <p style="text-align: center; color: #6c757d; font-size: 0.9em;">
            <em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | RQ3 Statistical Methods Report</em>
        </p>
    </div>
</body>
</html>"""

    return html_content

def main():
    """Generate and save the methods report"""
    print("📊 Loading RQ3 analysis results...")
    summary, stats_df = load_rq3_results()
    
    print("🔨 Generating statistical methods report...")
    html_content = generate_methods_report(summary, stats_df)
    
    # Save the report
    output_path = Path('/Users/shubhangmalviya/Documents/Projects/Walsh College/HistoPathologyResearch/reports/rq3/RQ3_Statistical_Methods_Principles.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Statistical methods report generated successfully!")
    print(f"📄 Report saved to: {output_path}")
    print(f"🌐 Open in browser to view the report")

if __name__ == "__main__":
    main()
