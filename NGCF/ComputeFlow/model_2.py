import graphviz
from graphviz import nohtml

g = graphviz.Digraph('NGCF_model5', format='png', node_attr={'shape': 'record'}, engine='dot')
g.attr(rankdir='TD')
with g.subgraph(name='SpMM') as c:
    c.node_attr.update(color='gray', style='filled', fillcolor='aquamarine', shape='Mrecord')
    c.node('spmm2', nohtml(r"copy_e_sum\n| {<in> in | graph | <out> out} | {one tensor | u\-\>i | \[M, d\]}"))
    # c.node('spmm3', nohtml(r"copy_e_sum\n| {<in> in | graph | <out> out} | {one tensor | i\-\>u | \[N, d\]}"))
    # c.node('spmm4', nohtml(r"copy_u\n| {<in> in| graph | <out> out} | {one tensor | i\-\>u | \[nnz, d\]}"))
    # c.node('spmm_u', nohtml(r"copy_u_sum\n| {<in> in | graph | <out> out} | {one tensor | self | \[N, d\]}"))
    c.node('spmm_i', nohtml(r"copy_u_sum\n| {<in> in | graph | <out> out} | {one tensor | self | \[M, d\]}"))


with g.subgraph(name='SDDMM') as c:
    c.node_attr.update(color='white', style='filled', fillcolor='darkturquoise', shape='Mrecord')
    c.node('sddmm1', nohtml(r"u_mul_v\n| {<in> in | graph | <out> out} | {two tensors | u\-i | \[nnz, d\]}"))
    c.node('sddmm2', nohtml(r"copy_u\n| {<in> in| graph |<out> out}  | {one tensor | u\-\>i |\[nnz, d\]}"))

    # c.node('sddmm2', nohtml(r"u_mul_v\n| {<in> in | graph | <out> out} | {two tensors | i\-u | \[nnz, d\]}"))


with g.subgraph(name='Linear') as c:
    c.node_attr.update(color='white', style='filled', fillcolor='darkgreen', fontcolor='white', shape='Mrecord')
    c.node('Linear1', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[M, d\]}"))
    c.node('Linear2', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[nnz, d\]}"))
    # c.node('Linear3', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[nnz, d\]}"))
    # c.node('Linear4', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[N, d\]}"))
    c.node('Linear5', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[N, d\]}"))
    # c.node('Linear6', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[M, d\]}"))

with g.subgraph(name='norm') as c:
    c.node_attr.update(color='gray', style='filled', fillcolor='azure2')
    c.node('norm_edge', nohtml(r"norm_edge\n| \[nnz, 1\]"))
    # c.node('norm_edge_1', nohtml(r"norm_edge\n| \[nnz, 1\]"))

with g.subgraph(name='in') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white')
    c.node('u_e', nohtml(r'user embedding | \[N, d\]'))
    # c.node('i_e', nohtml(r'item embedding | \[M, d\]'))
    # c.node('u_e_1', nohtml(r'user embedding | \[N, d\]'))
    c.node('i_e_1', nohtml(r'item embedding | \[M, d\]'))
    # c.node('u_e_2', nohtml(r'user embedding | \[N, d\]'))
    # c.node('i_e_2', nohtml(r'item embedding | \[M, d\]'))

with g.subgraph(name='weight') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='diamond')
    c.node('weight1', "W1")
    # c.node('weight1_1', "W1")
    c.node('weight1_2', "W1")
    # c.node('weight1_3', "W1")
    c.node('weight2', "W2")
    # c.node('weight2_1', "W2")

with g.subgraph(name='bias') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='polygon')
    c.node('b1', "b1")
    # c.node('b1_1', "b1")
    c.node('b1_2', "b1")
    # c.node('b1_3', "b1")
    c.node('b2', "b2")
    # c.node('b2_1', "b2")
    

with g.subgraph(name='operation') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='circle')
    c.node('mul1', nohtml(r'\*'))
    # c.node('mul2', nohtml(r'\*'))
    c.node('add1', nohtml(r'\+'))
    # c.node('add2', nohtml(r'\+'))
    c.node('add3', nohtml(r'\+'))
    # c.node('add4', nohtml(r'\+'))


with g.subgraph(name='in') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white')
    # c.node('u_e_o', nohtml(r'user embedding\' | \[N, d\]'))
    c.node('i_e_o', nohtml(r'item embedding\' | \[M, d\]'))

g.edge('i_e_1', 'Linear5:in')
# g.edge('i_e', 'Linear6:in')
g.edge('Linear5:out', 'sddmm2:in')
# g.edge('Linear6:out', 'spmm4:in')
g.edge('sddmm2:out', 'add3')
# g.edge('spmm4:out', 'add4')

g.edge('Linear2:out', 'add3')
# g.edge('Linear3:out', 'add4')
g.edge('add3', 'mul1')
# g.edge('add4', 'mul2')

g.edge('weight1_2', 'Linear5:in')
g.edge('b1_2', 'Linear5:in')
# g.edge('weight1_3', 'Linear6:in')
# g.edge('b1_3', 'Linear6:in')


g.edge('weight1', 'Linear1:in')
g.edge('b1', 'Linear1:in')

g.edge('u_e', 'Linear1:in')
# g.edge('i_e', 'Linear4:in')
g.edge('Linear1:out', 'spmm_i')
# g.edge('Linear4:out', 'spmm_u')

g.edge('weight2', 'Linear2:in')
g.edge('b2', 'Linear2:in')


# g.edge('weight2_1', 'Linear3:in')
# g.edge('b2_1', 'Linear3:in')
# g.edge('weight1_1', 'Linear4:in')
# g.edge('b1_1', 'Linear4:in')

g.edges([('u_e', 'sddmm1:in'),  ('i_e_1', 'sddmm1:in')])
# g.edges([('u_e_2', 'sddmm2:in'),  ('i_e', 'sddmm2:in')])

g.edge('norm_edge', 'mul1')
# g.edge('norm_edge_1', 'mul2')

g.edge('spmm2:out', 'add1')
# g.edge('spmm3:out', 'add2')
g.edge('add1', 'i_e_o')
# g.edge('add2', 'u_e_o')

g.edge('spmm_i:out', 'add1')
# g.edge('spmm_u:out', 'add2')

g.edge('sddmm1:out', 'Linear2:in')
# g.edge('sddmm2:out', 'Linear3:in')

g.edge('mul1', 'spmm2')
# g.edge('mul2', 'spmm3')


g.render(filename='model_2', directory="./",view=False)
