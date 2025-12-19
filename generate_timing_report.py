import json
import glob
import os
from datetime import datetime

def load_plans(base_dir):
    plans = []
    pattern = os.path.join(base_dir, "**", "plan_*_improved.json")
    files = glob.glob(pattern, recursive=True)
    
    for f in files:
        with open(f, 'r') as file:
            try:
                data = json.load(file)
                plans.append(data)
            except json.JSONDecodeError:
                print(f"Error reading {f}")
    return plans

def get_los_color(los):
    colors = {
        'A': '#4CAF50', # Green
        'B': '#8BC34A', # Light Green
        'C': '#CDDC39', # Lime
        'D': '#FFEB3B', # Yellow
        'E': '#FFC107', # Amber
        'F': '#F44336'  # Red
    }
    return colors.get(los, '#9E9E9E')

def generate_html(plans):
    # Group by intersection
    intersections = {}
    for plan in plans:
        iid = plan.get('intersection_id', 'Unknown')
        if iid not in intersections:
            intersections[iid] = []
        intersections[iid].append(plan)
    
    # Sort intersections
    sorted_ids = sorted(intersections.keys())

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Signal Timing Improvements Report</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; color: #333; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 40px; box-shadow: 0 0 20px rgba(0,0,0,0.1); border-radius: 8px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 40px; border-left: 5px solid #3498db; padding-left: 15px; }
            h3 { color: #7f8c8d; margin-top: 25px; }
            .plan-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 20px; margin-bottom: 30px; background-color: #fff; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .metric-box { padding: 15px; border-radius: 6px; text-align: center; color: white; }
            .metric-label { font-size: 0.9em; opacity: 0.9; margin-bottom: 5px; }
            .metric-value { font-size: 1.8em; font-weight: bold; }
            .metric-sub { font-size: 0.8em; opacity: 0.8; }
            
            table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 0.95em; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }
            th { background-color: #f8f9fa; font-weight: 600; color: #555; }
            tr:hover { background-color: #f9f9f9; }
            
            .diff-positive { color: #27ae60; font-weight: bold; }
            .diff-negative { color: #c0392b; font-weight: bold; }
            .diff-neutral { color: #7f8c8d; }
            
            .summary-stats { display: flex; gap: 20px; margin-bottom: 20px; background: #f8f9fa; padding: 15px; border-radius: 6px; }
            .stat-item { flex: 1; text-align: center; }
            .stat-val { font-size: 1.4em; font-weight: bold; color: #2c3e50; }
            .stat-lbl { font-size: 0.85em; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
            
            .los-badge { display: inline-block; width: 30px; height: 30px; line-height: 30px; text-align: center; border-radius: 50%; color: white; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Traffic Signal Timing Optimization Report</h1>
            <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
    """

    for iid in sorted_ids:
        html += f"<h2>Intersection: {iid}</h2>"
        
        # Sort plans by plan number
        intersection_plans = sorted(intersections[iid], key=lambda x: x.get('plan_number', 0))
        
        for plan in intersection_plans:
            curr = plan['current_timing']
            imp = plan['improved_timing']
            improv = plan['improvement']
            
            curr_los = curr.get('los', 'N/A')
            imp_los = imp.get('los', 'N/A')
            
            delay_red = improv.get('delay_reduction_pct', 0)
            delay_red_abs = improv.get('delay_reduction_s', 0)
            
            html += f"""
            <div class="plan-card">
                <h3>Plan {plan.get('plan_number')} - {plan.get('plan_name')} ({plan.get('time_range')})</h3>
                
                <div class="metrics-grid">
                    <div class="metric-box" style="background-color: {get_los_color(curr_los)}">
                        <div class="metric-label">Current LOS</div>
                        <div class="metric-value">{curr_los}</div>
                        <div class="metric-sub">{curr.get('delay', 0):.1f}s Delay</div>
                    </div>
                    <div class="metric-box" style="background-color: {get_los_color(imp_los)}">
                        <div class="metric-label">Improved LOS</div>
                        <div class="metric-value">{imp_los}</div>
                        <div class="metric-sub">{imp.get('delay', 0):.1f}s Delay</div>
                    </div>
                    <div class="metric-box" style="background-color: #3498db">
                        <div class="metric-label">Improvement</div>
                        <div class="metric-value">{delay_red:.1f}%</div>
                        <div class="metric-sub">-{delay_red_abs:.1f}s / veh</div>
                    </div>
                </div>
                
                <div class="summary-stats">
                    <div class="stat-item">
                        <div class="stat-val">{curr.get('cycle_length')}s &rarr; {imp.get('cycle_length')}s</div>
                        <div class="stat-lbl">Cycle Length</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-val">{improv.get('search_iterations')}</div>
                        <div class="stat-lbl">Optimization Iterations</div>
                    </div>
                </div>

                <table>
                    <thead>
                        <tr>
                            <th>Phase</th>
                            <th>Current Split (s)</th>
                            <th>New Split (s)</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # Combine keys from both
            all_phases = sorted(list(set(list(curr['phase_greens'].keys()) + list(imp['phase_greens'].keys()))))
            
            for phase in all_phases:
                c_val = curr['phase_greens'].get(phase, 0)
                i_val = imp['phase_greens'].get(phase, 0)
                diff = i_val - c_val
                
                diff_class = "diff-neutral"
                diff_str = "0"
                if diff > 0:
                    diff_class = "diff-positive"
                    diff_str = f"+{diff:.1f}"
                elif diff < 0:
                    diff_class = "diff-negative"
                    diff_str = f"{diff:.1f}"
                
                html += f"""
                        <tr>
                            <td>{phase}</td>
                            <td>{c_val:.1f}</td>
                            <td>{i_val:.1f}</td>
                            <td class="{diff_class}">{diff_str}</td>
                        </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """

    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    base_dir = "data/improved_timings"
    output_file = "timing_improvement_report.html"
    
    print(f"Scanning {base_dir}...")
    plans = load_plans(base_dir)
    print(f"Found {len(plans)} plans.")
    
    print("Generating HTML report...")
    html_content = generate_html(plans)
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
