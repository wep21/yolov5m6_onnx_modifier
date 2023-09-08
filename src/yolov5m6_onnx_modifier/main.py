import click
import onnx
import onnx_graphsurgeon as gs


@click.command()
@click.option("--input", help="input onnx file", default="input.onnx")
@click.option("--output", help="output onnx file", default="output.onnx")
def main(input, output):
    graph = gs.import_onnx(onnx.load(input))
    conv0 = [node for node in graph.nodes if node.name == "/model.33/m.0/Conv"][0]
    conv1 = [node for node in graph.nodes if node.name == "/model.33/m.1/Conv"][0]
    conv2 = [node for node in graph.nodes if node.name == "/model.33/m.2/Conv"][0]
    conv3 = [node for node in graph.nodes if node.name == "/model.33/m.3/Conv"][0]
    for conv in [conv0, conv1, conv2, conv3]:
        conv.o().inputs.clear()
        conv.outputs[0].dtype = graph.inputs[0].dtype
    graph.outputs = [
        conv0.outputs[0],
        conv1.outputs[0],
        conv2.outputs[0],
        conv3.outputs[0]
    ]
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), output)

if __name__ == '__main__':
    main()
