"""Distinct colors for every released CCF label, stable across sections/cuts."""
import json, colorsys

def distinct_references(refs, graph=None):
    ids=sorted({str(i) for info in refs.values() for i in info['ids']})
    assigned={};used={'#dddddd'};slot=0
    for label in ids:
        while True:
            rgb=colorsys.hsv_to_rgb((.13+slot*.61803398875)%1,[.68,.84,.95][slot%3],[.80,.94][slot%2])
            color='#%02x%02x%02x'%tuple(round(255*x) for x in rgb);slot+=1
            if color not in used:break
        assigned[label]=color;used.add(color)
    result=json.loads(json.dumps(refs))
    for info in result.values():
        info['colors']=[assigned[str(i)] for i in info['ids']]
        if len(set(info['colors']))!=info['k']:raise ValueError('Nonunique reference palette')
    return result
