"""Public0.6.1 measurement code; change duration stop to12 cold samples.
Unique upstream prompt prefixes retained; persist each sample for audit.
No change to payload, timing, tokenizer, prefill arithmetic or aggregation.
"""
import pathlib,sys,hashlib
source=pathlib.Path('/tmp/lab-community.py')
s=source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest()=='516dc590b80dda6d9880f9f5034026fdf1b2074e9747bf8a920758680f019817'
old='if elapsed >= PREFILL_DURATION and (len(prefill_samples) >= 2 or r >= 1):'
assert s.count(old)==1
s=s.replace(old,'if len(prefill_samples) >= 12:')
old='                        r += 1\n                        live.update(build_display(state))'
assert s.count(old)==1
s=s.replace(old,'''                        with open(args.output + ".samples.jsonl", "a") as sample_file:
                            sample_file.write(json.dumps({"ctx": ctx, "iteration": r, "sample": sample}) + "\\n")
                        r += 1
                        live.update(build_display(state))''')
__file__=str(source)
exec(compile(s,str(source),'exec'),globals())
