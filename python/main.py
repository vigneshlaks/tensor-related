import json

def ir_to_json():
    pass

def json_to_ir(json):
    lines = []
    for instr in json["instructions"]:
        string = None
        if instr["op"] == "const":
            string = f"{instr["dest"]} = {instr["op"]} {instr["value"]};"
        else:
            string = f"{instr["dest"]} = {instr["op"]} {instr["args"]};"
        lines.append(string)

    with open("output.txt", "w") as file:
        file.write("\n".join(lines))

'''
Then we need to write dead code elimination

Then need to write ssa?

Need to also write operator fusion.
'''

if __name__ == "__main__":
    with open("data.json", 'r') as file:
        data = json.load(file)
    
    json_to_ir(data)
    