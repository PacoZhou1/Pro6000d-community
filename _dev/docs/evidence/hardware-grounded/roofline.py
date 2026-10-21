"""Ideal single-GEMM DRAM floor, not a measured GPU roofline or API prediction."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BANDWIDTH=1792e9
FP8=1007.6e12
BF16=503.8e12
rows=[]
for name,n,k,dtype,peak,ms in [('gate_up',34816,5120,'fp8',FP8,[1,5,2741,16160]),
                              ('down',5120,17408,'fp8',FP8,[1,5,2741,16160]),
                              ('lm_head',248320,5120,'bf16',BF16,[1,5])]:
 for m in ms:
  bpe=1 if dtype=='fp8' else 2
  a=m*k*bpe;b=n*k*bpe;d=m*n*2
  scales=m*(k//128)*4+(n//128)*(k//128)*4 if dtype=='fp8' else 0
  volume=a+b+d+scales
  flops=2*m*n*k
  rows.append({'operator':name,'M':m,'N':n,'K':k,'dtype':dtype,
    'flops':flops,'unique_A_bytes':a,'unique_B_bytes':b,'BF16_D_write_bytes':d,
    'scale_bytes':scales,'ideal_one_read_write_bytes':volume,
    'ideal_arithmetic_intensity_FLOP_per_byte':flops/volume,
    'advertised_dense_FP32_acc_peak_TFLOPs':peak/1e12,
    'ideal_compute_floor_ms':flops/peak*1000,
    'ideal_dram_floor_ms':volume/BANDWIDTH*1000,
    'ideal_roofline_floor_ms':max(flops/peak,volume/BANDWIDTH)*1000})
source=ROOT.parent/'gemm-targeted/module-signatures.json'
# Require these dimensions to have occurred in exact module-attributed real traces.
signatures=json.loads(source.read_text())['records']
for r in rows:
 assert any(x['M']==r['M'] and x['N']==r['N'] and x['K']==r['K'] for x in signatures)
result={'source_shape_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'peak_source':'NVIDIA RTX PRO Blackwell GPU Architecture whitepaper v1.1 Table 4; exact URL and primary scope in sm120-facts.md',
 'memory_bandwidth_GB_s':BANDWIDTH/1e9,'FP8_ridge_FLOP_per_byte':FP8/BANDWIDTH,
 'BF16_ridge_FLOP_per_byte':BF16/BANDWIDTH,
 'limits':['Conditional uncached one-read/write traffic and advertised boost-clock dense peaks, not measured sustainable roofline; cached operands can have less DRAM traffic.',
 'Does not model repeated L2/shared traffic, instruction/scale conversion overhead, clock variation, embedding contention, or API scheduling.',
 'BF16_D is one GEMM output write; complete activation-quant pipeline is a separate measurement.',
 'BF16 lm_head is attribution only; changing its precision is outside scope.'], 'cases':rows}
(ROOT/'roofline.json').write_text(json.dumps(result,indent=2)+'\n')
for r in rows:
 print(r['operator'],r['M'],'AI',round(r['ideal_arithmetic_intensity_FLOP_per_byte'],2),'compute_ms',round(r['ideal_compute_floor_ms'],4),'dram_ms',round(r['ideal_dram_floor_ms'],4))
