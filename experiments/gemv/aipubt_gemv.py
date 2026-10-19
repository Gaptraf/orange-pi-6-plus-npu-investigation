"""Isolated custom GEMV plugin; only matches our private experimental op type."""
from AIPUBuilder.core import BuilderOpPlugin, register_optype, BuilderParams, DataLayout
from AIPUBuilder.plugin_loader import register_plugin, PluginType
op_type=register_optype('Opi6PackedGemv')
@register_plugin(PluginType.Builder,0)
class PackedGemv(BuilderOpPlugin):
    def get_graph_pattern(self): return ([('gemv',op_type)],[])
    def get_score(self): return 100
    def set_target(self,target): return True
    def check_params(self,nodes): return True
    def setup(self,sgnode,nodes):
        sgnode.attrs['keeping_layout']=False
        return True
    def generate_code_name(self,sgnode,nodes): return 'gemv_packed.o'
    def generate_kernel(self,sgnode,nodes): return 'gemv_packed'
    def generate_params(self,sgnode,nodes):
        ro=BuilderParams();name=nodes[0].name
        for value in [sgnode.inputs[0],sgnode.constants[name+'/wg'],sgnode.constants[name+'/wu'],
                      sgnode.outputs[0],sgnode.outputs[1],nodes[0].params['k4'],nodes[0].params['n32']]:
            ro.append(value)
        return ro
