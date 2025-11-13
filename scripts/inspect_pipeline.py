import sys, inspect
sys.path.insert(0, '.')
from inference.pipeline import InferencePipeline

print('--- InferencePipeline.__init__ signature ---')
print(inspect.signature(InferencePipeline.__init__))

print('\\n--- Small class docstring (if any) ---')
print(InferencePipeline.__doc__ or 'No docstring')

print('\\n--- First 120 lines of source ---')
try:
    src = inspect.getsource(InferencePipeline)
    for i, line in enumerate(src.splitlines()[:120], 1):
        print(f'{i:03d}: {line}')
except Exception as e:
    print('Could not show source:', e)
