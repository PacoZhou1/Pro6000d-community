"""Exercise actual fixed-stack FX Node and OpOverload serialization on CPU."""
import json
import torch
from fx_probe import _safe_arg

graph = torch.fx.Graph()
node = graph.placeholder('input')
values = [node, torch.ops.aten.add.Tensor, None, 128, True, [node, torch.ops.aten.t.default]]
records = [_safe_arg(value) for value in values]
assert records[0] == {'node': node.name}
assert records[1] == {'operator': 'aten.add.Tensor'}
assert records[-1] == [{'node': node.name}, {'operator': 'aten.t.default'}]
print(json.dumps({'passed': True, 'torch': torch.__version__, 'serialized': records}))
