"""Distinct CCF colors; anatomical IDs and metric labels are unchanged."""
import json, colorsys

def distinct_references(refs, graph):
    """One stable RGB color per structure ID across this report's resolutions.

    Ontology traversal groups neighboring anatomical labels. Golden-angle hue
    steps avoid assigning a single inherited hue to the layers of an area.
    No color assignment uses model clusters, scores, or expression values.
    """
    nodes = {}

    def walk(node):
        nodes[int(node['id'])] = node
        for child in node.get('children', []):
            walk(child)

    for root in graph['msg']:
        walk(root)
    ids = sorted({int(i) for info in refs.values() for i in info['ids']},
                 key=lambda i: (nodes[i]['graph_order'], i))
    assigned = {}
    used = {'#dddddd'}  # Grey remains reserved for unavailable labels.
    slot = 0
    for structure in ids:
        while True:
            rgb = colorsys.hsv_to_rgb((.13 + slot * .61803398875) % 1,
                                     [.68, .84, .95][slot % 3],
                                     [.80, .94][slot % 2])
            color = '#%02x%02x%02x' % tuple(round(255 * x) for x in rgb)
            slot += 1
            if color not in used:
                break
        assigned[structure] = color
        used.add(color)
    result = json.loads(json.dumps(refs))
    for level, info in result.items():
        info['colors'] = [assigned[int(i)] for i in info['ids']]
        if len(set(info['colors'])) != info['k']:
            raise ValueError('Color uniqueness check failed: ' + level)
    return result
