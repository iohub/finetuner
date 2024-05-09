import sys, json


'''
{"instruction_zh": "编辑以下句子以提高清晰度和流畅性。", "input_zh": "这家餐厅的食物很棒，尽管价格有点贵。", "output_zh": "虽然这家餐厅的食物很好，但价格有点贵。" }

->

{
    "prompt": "instruction_zh + input_zh",
    "completion": "output_zh"
}

'''

infile = sys.argv[1]
outfile = sys.argv[2]
N = int(sys.argv[3])

corpus = []
with open(infile, 'r') as fobj:
    for line in fobj:
        jdata = json.loads(line.strip())
        prompt = jdata['instruction_zh'] + '\ninput:' + jdata['input_zh'] if jdata['input_zh'] else jdata['instruction_zh']
        corpus.append({
            'prompt': prompt,
            'completion': jdata['output_zh']
        })
        if N != -1 and len(corpus) > N: break

print('total enties:', len(corpus))

with open(outfile, 'w') as fobj:
    fobj.write(json.dumps(corpus, indent=2, ensure_ascii=False))