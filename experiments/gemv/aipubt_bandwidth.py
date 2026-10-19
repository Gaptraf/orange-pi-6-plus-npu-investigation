"""Private op for the bounded mixed-engine experiment."""
from AIPUBuilder.core import BuilderOpPlugin, register_optype, BuilderParams
from AIPUBuilder.plugin_loader import register_plugin, PluginType
op_type=register_optype('Opi6BandwidthDma')
@register_plugin(PluginType.Builder,0)
class BandwidthDma(BuilderOpPlugin):
    def get_graph_pattern(self):return ([('dma',op_type)],[])
    def get_score(self):return 100
    def set_target(self,target):return True
    def check_params(self,nodes):return True
    def setup(self,sgnode,nodes):
        sgnode.attrs['keeping_layout']=False
        return True
    def generate_code_name(self,sgnode,nodes):return 'bandwidth_dma.o'
    def generate_kernel(self,sgnode,nodes):return 'bandwidth_dma'
    def generate_params(self,sgnode,nodes):
        params=BuilderParams()
        for value in [sgnode.inputs[0],sgnode.constants[nodes[0].name+'/stream'],
                      sgnode.outputs[0],nodes[0].params['chunks']]:params.append(value)
        return params
