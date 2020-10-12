from pathlib import Path
import tempfile
from transformers.convert_graph_to_onnx import convert, quantize

dest = Path(tempfile.mkdtemp(), "text-generation.onnx")
convert(
  pipeline_name="text-generation",
  model="gpt2",
  output=dest,
  framework="pt",
  opset=11
)
quantize(dest)
