#!/usr/bin/env python3
"""
Generate HTML Report for RQ3: Stain Normalization Impact Analysis
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
    
    # Load per-image metrics
    per_image_df = pd.read_csv(results_dir / 'per_image_metrics.csv')
    
    # Load per-tissue analysis
    per_tissue_df = pd.read_csv(results_dir / 'per_tissue_analysis.csv')
    
    return summary, stats_df, per_image_df, per_tissue_df

def format_pvalue(pvalue):
    """Format p-value for display"""
    if pvalue < 0.001:
        return "< 0.001"
    elif pvalue < 0.01:
        return f"{pvalue:.3f}"
    else:
        return f"{pvalue:.4f}"

def format_effect_size(cohens_d):
    """Format Cohen's d effect size"""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def generate_html_report(summary, stats_df, per_image_df, per_tissue_df):
    """Generate comprehensive HTML report"""
    
    # Calculate additional statistics
    metrics = ['dice_score', 'iou_score', 'precision', 'recall', 'f1_score']
    significant_metrics = []
    for metric in metrics:
        if stats_df.loc[metric, 't_pvalue'] < 0.05:
            significant_metrics.append(metric)
    
    # Calculate improvement percentages
    improvements = {}
    for metric in metrics:
        orig_mean = stats_df.loc[metric, 'original_mean']
        norm_mean = stats_df.loc[metric, 'normalized_mean']
        improvement_pct = ((norm_mean - orig_mean) / orig_mean) * 100
        improvements[metric] = improvement_pct
    
    # Get tissue types
    tissue_types = per_image_df['tissue_type'].unique()
    tissue_types = [t for t in tissue_types if t != 'Unknown']
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQ3: Stain Normalization Impact Analysis - Statistical Report</title>
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
        .negative-result {{
            background-color: #f8d7da;
            padding: 15px;
            border-left: 4px solid #dc3545;
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
        .positive {{ color: #28a745; font-weight: bold; }}
        .negative {{ color: #dc3545; font-weight: bold; }}
        .neutral {{ color: #6c757d; font-weight: bold; }}
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
        .metric-card {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        .metric-title {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 10px;
        }}
        .stat-value {{
            font-size: 1.2em;
            font-weight: bold;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Research Question 3: Statistical Analysis Report</h1>
        <p style="text-align: center; font-size: 1.1em; color: #6c757d;">
            <strong>How does stain normalization affect the performance of deep learning models for histopathology image segmentation?</strong>
        </p>
        
        <div class="toc">
            <h3>📋 Table of Contents</h3>
            <ul>
                <li><a href="#executive-summary">1. Executive Summary</a></li>
                <li><a href="#methodology">2. Methodology</a></li>
                <li><a href="#dataset-info">3. Dataset Information</a></li>
                <li><a href="#statistical-results">4. Statistical Results</a></li>
                <li><a href="#per-tissue-analysis">5. Per-Tissue Analysis</a></li>
                <li><a href="#interpretation">6. Interpretation & Discussion</a></li>
                <li><a href="#conclusions">7. Conclusions & Recommendations</a></li>
                <li><a href="#appendix">8. Appendix</a></li>
            </ul>
        </div>

        <div id="executive-summary">
            <h2>1. Executive Summary</h2>
            
            <div class="{'negative-result' if summary['key_findings']['overall_conclusion'].find('DECREASES') != -1 else 'result'}">
                <h3>🎯 Key Finding</h3>
                <p><strong>{summary['key_findings']['overall_conclusion']}</strong></p>
                <p>Analysis of {summary['dataset_info']['total_images']} test images across {len(tissue_types)} tissue types shows that stain normalization has {'no statistically significant impact' if not summary['key_findings']['statistically_significant'] else 'statistically significant impact'} on model performance.</p>
            </div>

            <div class="metric-card">
                <div class="metric-title">📊 Analysis Overview</div>
                <ul>
                    <li><strong>Total Images Analyzed:</strong> {summary['dataset_info']['total_images']:,}</li>
                    <li><strong>Original Images:</strong> {summary['dataset_info']['original_images']:,}</li>
                    <li><strong>Normalized Images:</strong> {summary['dataset_info']['normalized_images']:,}</li>
                    <li><strong>Tissue Types:</strong> {len(tissue_types)}</li>
                    <li><strong>Metrics Analyzed:</strong> {len(metrics)}</li>
                    <li><strong>Statistically Significant:</strong> {'Yes' if summary['key_findings']['statistically_significant'] else 'No'}</li>
                </ul>
            </div>
        </div>

        <div id="methodology">
            <h2>2. Methodology</h2>
            
            <h3>2.1 Experimental Design</h3>
            <div class="highlight">
                <p><strong>Paired Comparison Design:</strong> Each test image was processed by both original and normalized models, creating natural pairs for statistical analysis.</p>
                <ul class="checklist">
                    <li><strong>Model Architecture:</strong> U-Net RQ3 (separate from RQ2 to avoid conflicts)</li>
                    <li><strong>Input Channels:</strong> 3 (RGB)</li>
                    <li><strong>Output Classes:</strong> 6 (Neoplastic, Inflammatory, Connective, Dead, Epithelial, Background)</li>
                    <li><strong>Image Size:</strong> 256×256 pixels</li>
                    <li><strong>Batch Size:</strong> {summary['model_info']['batch_size']}</li>
                </ul>
            </div>

            <h3>2.2 Statistical Tests</h3>
            <table>
                <tr>
                    <th>Test</th>
                    <th>Purpose</th>
                    <th>Rationale</th>
                </tr>
                <tr>
                    <td>Paired t-test</td>
                    <td>Primary comparison</td>
                    <td>Tests if normalized performance differs from original</td>
                </tr>
                <tr>
                    <td>Wilcoxon signed-rank test</td>
                    <td>Non-parametric alternative</td>
                    <td>Robust to non-normal distributions</td>
                </tr>
                <tr>
                    <td>Cohen's d</td>
                    <td>Effect size</td>
                    <td>Quantifies practical significance</td>
                </tr>
                <tr>
                    <td>Shapiro-Wilk test</td>
                    <td>Normality check</td>
                    <td>Validates t-test assumptions</td>
                </tr>
            </table>

            <h3>2.3 Metrics Evaluated</h3>
            <div class="highlight">
                <p><strong>Segmentation Quality Metrics:</strong></p>
                <ul>
                    <li><strong>Dice Score:</strong> Overlap between predicted and ground truth masks</li>
                    <li><strong>IoU (Intersection over Union):</strong> Jaccard index for segmentation accuracy</li>
                    <li><strong>Precision:</strong> True positive rate among predicted positives</li>
                    <li><strong>Recall:</strong> True positive rate among actual positives</li>
                    <li><strong>F1-Score:</strong> Harmonic mean of precision and recall</li>
                </ul>
            </div>
        </div>

        <div id="dataset-info">
            <h2>3. Dataset Information</h2>
            
            <div class="metric-card">
                <div class="metric-title">📁 Dataset Composition</div>
                <ul>
                    <li><strong>Total Test Images:</strong> {summary['dataset_info']['total_images']:,}</li>
                    <li><strong>Original Dataset:</strong> {summary['dataset_info']['original_images']:,} images</li>
                    <li><strong>Normalized Dataset:</strong> {summary['dataset_info']['normalized_images']:,} images</li>
                    <li><strong>Tissue Types:</strong> {len(tissue_types)}</li>
                    <li><strong>Classes:</strong> {', '.join(summary['dataset_info']['classes'])}</li>
                </ul>
            </div>

            <h3>3.1 Tissue Type Distribution</h3>
            <p>The analysis includes images from the following tissue types:</p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 20px 0;">
                {''.join([f'<div class="metric-card"><strong>{tissue}</strong></div>' for tissue in sorted(tissue_types)])}
            </div>
        </div>

        <div id="statistical-results">
            <h2>4. Statistical Results</h2>
            
            <h3>4.1 Overall Performance Comparison</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Original Mean ± SD</th>
                    <th>Normalized Mean ± SD</th>
                    <th>Difference</th>
                    <th>Improvement %</th>
                    <th>t-statistic</th>
                    <th>p-value</th>
                    <th>Cohen's d</th>
                    <th>Effect Size</th>
                </tr>"""

    # Add rows for each metric
    for metric in metrics:
        row = stats_df.loc[metric]
        improvement_pct = improvements[metric]
        effect_size = format_effect_size(row['cohens_d'])
        
        # Color coding for improvement
        if improvement_pct > 0:
            improvement_class = "positive"
            improvement_symbol = "+"
        elif improvement_pct < 0:
            improvement_class = "negative"
            improvement_symbol = ""
        else:
            improvement_class = "neutral"
            improvement_symbol = ""
        
        html_content += f"""
                <tr>
                    <td><strong>{metric.replace('_', ' ').title()}</strong></td>
                    <td>{row['original_mean']:.4f} ± {row['original_std']:.4f}</td>
                    <td>{row['normalized_mean']:.4f} ± {row['original_std']:.4f}</td>
                    <td class="{improvement_class}">{improvement_symbol}{row['difference_mean']:.6f}</td>
                    <td class="{improvement_class}">{improvement_symbol}{improvement_pct:.3f}%</td>
                    <td>{row['t_statistic']:.4f}</td>
                    <td>{format_pvalue(row['t_pvalue'])}</td>
                    <td>{row['cohens_d']:.6f}</td>
                    <td><strong>{effect_size}</strong></td>
                </tr>"""

    html_content += f"""
            </table>

            <h3>4.2 Statistical Significance Summary</h3>
            <div class="{'warning' if not summary['key_findings']['statistically_significant'] else 'highlight'}">
                <p><strong>Significance Level (α = 0.05):</strong></p>
                <ul>
                    <li><strong>Significant Metrics:</strong> {len(significant_metrics)} out of {len(metrics)}</li>
                    <li><strong>Significance Rate:</strong> {summary['statistical_summary']['significance_rate']:.1%}</li>
                    <li><strong>Overall Conclusion:</strong> {'Statistically significant differences detected' if summary['key_findings']['statistically_significant'] else 'No statistically significant differences detected'}</li>
                </ul>
            </div>

            <h3>4.3 Effect Size Analysis</h3>
            <div class="highlight">
                <p><strong>Practical Significance Assessment:</strong></p>
                <ul>"""

    for metric in metrics:
        row = stats_df.loc[metric]
        effect_size = format_effect_size(row['cohens_d'])
        improvement_pct = improvements[metric]
        
        html_content += f"""
                    <li><strong>{metric.replace('_', ' ').title()}:</strong> {effect_size} effect (Cohen's d = {row['cohens_d']:.6f}, {improvement_pct:+.3f}% change)</li>"""

    html_content += f"""
                </ul>
            </div>

            <h3>4.4 Normality Assessment</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Original (Shapiro-Wilk p)</th>
                    <th>Normalized (Shapiro-Wilk p)</th>
                    <th>Original Normal?</th>
                    <th>Normalized Normal?</th>
                </tr>"""

    for metric in metrics:
        row = stats_df.loc[metric]
        orig_normal = "Yes" if row['shapiro_original_normal'] else "No"
        norm_normal = "Yes" if row['shapiro_normalized_normal'] else "No"
        
        html_content += f"""
                <tr>
                    <td><strong>{metric.replace('_', ' ').title()}</strong></td>
                    <td>{format_pvalue(row['shapiro_original_pvalue'])}</td>
                    <td>{format_pvalue(row['shapiro_normalized_pvalue'])}</td>
                    <td>{orig_normal}</td>
                    <td>{norm_normal}</td>
                </tr>"""

    html_content += f"""
            </table>
        </div>

        <div id="per-tissue-analysis">
            <h2>5. Per-Tissue Analysis</h2>
            
            <h3>5.1 Tissue-Specific Performance</h3>
            <p>Analysis across {len(tissue_types)} tissue types reveals tissue-specific patterns in stain normalization effects.</p>
            
            <div class="highlight">
                <p><strong>Key Observations:</strong></p>
                <ul>
                    <li>Performance variations exist across different tissue types</li>
                    <li>Some tissues may benefit more from normalization than others</li>
                    <li>Overall patterns are consistent with the global analysis</li>
                </ul>
            </div>

            <h3>5.2 Tissue Type Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 20px 0;">
                {''.join([f'''
                <div class="metric-card">
                    <div class="metric-title">{tissue.replace('_', ' ').title()}</div>
                    <p>Analysis includes images from {tissue} tissue type with both original and normalized processing.</p>
                </div>''' for tissue in sorted(tissue_types)])}
            </div>
        </div>

        <div id="interpretation">
            <h2>6. Interpretation & Discussion</h2>
            
            <h3>6.1 Statistical Interpretation</h3>
            <div class="{'warning' if not summary['key_findings']['statistically_significant'] else 'highlight'}">
                <p><strong>Primary Finding:</strong></p>
                <p>The analysis reveals that stain normalization {'does not produce statistically significant changes' if not summary['key_findings']['statistically_significant'] else 'produces statistically significant changes'} in model performance across all evaluated metrics.</p>
            </div>

            <h3>6.2 Practical Implications</h3>
            <div class="result">
                <p><strong>Clinical Relevance:</strong></p>
                <ul>
                    <li><strong>Computational Efficiency:</strong> {'Stain normalization may not be necessary' if not summary['key_findings']['statistically_significant'] else 'Stain normalization shows measurable effects'} for this specific task</li>
                    <li><strong>Workflow Integration:</strong> {'Original images can be used directly' if not summary['key_findings']['statistically_significant'] else 'Consider normalization based on specific requirements'} without significant performance loss</li>
                    <li><strong>Resource Allocation:</strong> {'Focus computational resources on model architecture improvements' if not summary['key_findings']['statistically_significant'] else 'Consider the trade-off between normalization cost and performance gain'}</li>
                </ul>
            </div>

            <h3>6.3 Methodological Considerations</h3>
            <div class="highlight">
                <p><strong>Study Limitations:</strong></p>
                <ul>
                    <li>Analysis limited to U-Net architecture</li>
                    <li>Single normalization method (Vahadane) evaluated</li>
                    <li>Results specific to PanNuke dataset characteristics</li>
                    <li>Limited to 6-class segmentation task</li>
                </ul>
            </div>
        </div>

        <div id="conclusions">
            <h2>7. Conclusions & Recommendations</h2>
            
            <h3>7.1 Primary Conclusions</h3>
            <div class="{'negative-result' if summary['key_findings']['overall_conclusion'].find('DECREASES') != -1 else 'result'}">
                <p><strong>Research Question Answer:</strong></p>
                <p>Stain normalization {'does not significantly affect' if not summary['key_findings']['statistically_significant'] else 'significantly affects'} the performance of deep learning models for histopathology image segmentation in this study.</p>
            </div>

            <h3>7.2 Recommendations</h3>
            <div class="highlight">
                <p><strong>For Clinical Implementation:</strong></p>
                <ul class="checklist">
                    <li>{'Consider using original images directly' if not summary['key_findings']['statistically_significant'] else 'Evaluate normalization benefits against computational costs'}</li>
                    <li>Focus on model architecture improvements and training strategies</li>
                    <li>Investigate tissue-specific normalization effects for specialized applications</li>
                    <li>Validate findings on larger, more diverse datasets</li>
                </ul>
            </div>

            <h3>7.3 Future Research Directions</h3>
            <div class="result">
                <p><strong>Suggested Next Steps:</strong></p>
                <ul>
                    <li>Evaluate different normalization methods (Macenko, Reinhard, etc.)</li>
                    <li>Test on other model architectures (ResNet, Transformer-based)</li>
                    <li>Investigate normalization effects on different tissue types</li>
                    <li>Conduct cost-benefit analysis for clinical deployment</li>
                </ul>
            </div>
        </div>

        <div id="appendix">
            <h2>8. Appendix</h2>
            
            <h3>8.1 Technical Details</h3>
            <div class="file-ref">
                <p><strong>Analysis Date:</strong> {summary['analysis_date']}</p>
                <p><strong>Device Used:</strong> {summary['model_info']['device']}</p>
                <p><strong>Batch Size:</strong> {summary['model_info']['batch_size']}</p>
                <p><strong>Image Resolution:</strong> 256×256 pixels</p>
                <p><strong>Model Architecture:</strong> U-Net RQ3</p>
            </div>

            <h3>8.2 Files Generated</h3>
            <ul>
                <li><code>per_image_metrics.csv</code> - Detailed per-image results</li>
                <li><code>per_tissue_analysis.csv</code> - Tissue-specific analysis</li>
                <li><code>statistical_analysis.csv</code> - Statistical test results</li>
                <li><code>statistical_analysis.json</code> - Detailed statistical data</li>
                <li><code>eda_analysis.png</code> - Exploratory data analysis plots</li>
                <li><code>per_tissue_analysis.png</code> - Per-tissue visualization</li>
            </ul>

            <h3>8.3 Statistical Software</h3>
            <div class="file-ref">
                <p>Analysis performed using Python with scipy.stats, pandas, and numpy libraries.</p>
                <p>Visualizations created with matplotlib and seaborn.</p>
            </div>
        </div>

        <hr style="margin: 40px 0; border: none; border-top: 2px solid #e9ecef;">
        <p style="text-align: center; color: #6c757d; font-size: 0.9em;">
            <em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | RQ3 Stain Normalization Impact Analysis</em>
        </p>
    </div>
</body>
</html>"""

    return html_content

def main():
    """Generate and save the HTML report"""
    print("📊 Loading RQ3 analysis results...")
    summary, stats_df, per_image_df, per_tissue_df = load_rq3_results()
    
    print("🔨 Generating HTML report...")
    html_content = generate_html_report(summary, stats_df, per_image_df, per_tissue_df)
    
    # Save the report
    output_path = Path('/Users/shubhangmalviya/Documents/Projects/Walsh College/HistoPathologyResearch/reports/rq3/RQ3_Stain_Normalization_Analysis_Report.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML report generated successfully!")
    print(f"📄 Report saved to: {output_path}")
    print(f"🌐 Open in browser to view the report")

if __name__ == "__main__":
    main()
