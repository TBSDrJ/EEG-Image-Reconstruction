import pathlib
from pprint import pprint

base_path = pathlib.Path("ridge_comparisons/saves")
values = []
for path in base_path.iterdir():
    
    if "printout" in path.name:
        with open(path, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "Failed to improve" in line:
                alg_code = path.name.split("_")[2]
                dist = lines[i-1][:-1].split()[4][:-1]
                corr = lines[i-1][:-1].split()[9][:7]
                # print(f"{alg_code=} {dist=} {corr=}")
                if dist != 'nan' and corr != 'nan':
                    values.append((alg_code, dist, corr))
values.sort(key=lambda x: x[0])
# pprint(values)
table_str = ""
i = 0
stop_checking = False
for loss in range(3):
    table_str += "<table>\n"
    table_str += "\t<tr>\n\t\t<td></td>\n"
    table_str += "\t\t<td colspan='6'><center><strong>Learning Rate</strong>"
    table_str += "</center></td>\n\t</tr>\n\t<tr>\n"
    table_str += "\t\t<td></td>\n\t\t<td>0.01</td>\n\t\t<td>0.001</td>\n"
    table_str += "\t\t<td>0.0001</td>\n\t\t<td>0.01<br/>with decay</td>\n"
    table_str += "\t\t<td>0.001<br/>with decay</td>\n"
    table_str += "\t\t<td>0.0001<br/>with decay</td>\n"
    if i == len(values):
        stop_checking = True
    for model in range(8):
        table_str += f"\t<tr> <!-- {loss}{model}x -->\n"
        table_str += f"\t\t<td><strong>Model {model}</strong></td>\n"
        if i == len(values): 
            stop_checking = True
        for lr in range(6):
            if not stop_checking and values[i][0] == f"{loss}{model}{lr}":
                cell_str = "\t\t<td><span style='color:"
                if float(values[i][1]) >= 113.37:
                    cell_str += "red"
                else:
                    cell_str += "green"
                cell_str += f"'>dist = {values[i][1]}</span><br/><span style='color:"
                if float(values[i][2]) > 0.023811:
                    cell_str += "green"
                else:
                    cell_str += "red"
                cell_str += f"'>corr = {values[i][2]}</td><!-- {values[i][0]} -->\n"
                i += 1
            else:
                cell_str = "\t\t<td></td>\n"
            table_str += cell_str
        table_str += "\t</tr>\n"
    table_str += "</table>\n"

with open('ridge_comparisons/tmp/readme_table.txt', 'w') as f:
    print(table_str, file=f)