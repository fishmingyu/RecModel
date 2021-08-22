import graphviz
from graphviz import nohtml

g = graphviz.Digraph('NGCF_model5', format='png', node_attr={'shape': 'record'}, engine='dot')
g.attr(rankdir='TD')
with g.subgraph(name='SpMM') as c:
    c.node_attr.update(color='gray', style='filled', fillcolor='aquamarine', shape='Mrecord')
    c.node('spmm1', nohtml(r"u_mul_e_sum\n| {<in> in| graph |<out> out}  | {two tensors | i\-\>u |\[N, d\]}"))
    c.node('spmm2', nohtml(r"copy_e_sum\n| {<in> in | graph | <out> out} | {one tensor | u\-\>i | \[M, d\]}"))
    c.node('spmm3', nohtml(r"copy_e_sum\n| {<in> in | graph | <out> out} | {one tensor | i\-\>u | \[N, d\]}"))
    c.node('spmm4', nohtml(r"u_mul_e_sum\n| {<in> in| graph | <out> out} | {two tensors | u\-\>i | \[M, d\]}"))


with g.subgraph(name='SDDMM') as c:
    c.node_attr.update(color='white', style='filled', fillcolor='darkturquoise', shape='Mrecord')
    c.node('sddmm', nohtml(r"u_mul_v\n| {<in> in | graph | <out> out} | {two tensors | u\-i | \[nnz, d\]}"))

with g.subgraph(name='Linear') as c:
    c.node_attr.update(color='white', style='filled', fillcolor='darkgreen', fontcolor='white', shape='Mrecord')
    c.node('Linear1', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[M, d\]}"))
    c.node('Linear2', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[M, d\]}"))
    c.node('Linear3', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[N, d\]}"))
    c.node('Linear4', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[N, d\]}"))

with g.subgraph(name='norm') as c:
    c.node_attr.update(color='gray', style='filled', fillcolor='azure2')
    c.node('norm_user', nohtml(r"norm_user\n| \[N, 1\]"))
    c.node('norm_item', nohtml(r"norm_item\n| \[M, 1\]"))
    c.node('norm_edge', nohtml(r"norm_edge\n| \[nnz, 1\]"))
    c.node('norm_edge_1', nohtml(r"norm_edge\n| \[nnz, 1\]"))

with g.subgraph(name='in') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white')
    c.node('u_e', nohtml(r'user embedding | \[N, d\]'))
    c.node('i_e', nohtml(r'item embedding | \[M, d\]'))
    c.node('u_e_1', nohtml(r'user embedding | \[N, d\]'))
    c.node('i_e_1', nohtml(r'item embedding | \[M, d\]'))

with g.subgraph(name='weight') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='diamond')
    c.node('weight1', "W1")
    c.node('weight1_1', "W1")
    c.node('weight2', "W2")
    c.node('weight2_1', "W2")

with g.subgraph(name='bias') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='polygon')
    c.node('b1', "b1")
    c.node('b1_1', "b1")
    c.node('b2', "b2")
    c.node('b2_1', "b2")

with g.subgraph(name='operation') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='ellipse')
    c.node('mul1', nohtml(r'element-wise\n multiplication'))
    c.node('mul2', nohtml(r'element-wise\n multiplication'))
    c.node('add1', nohtml(r'\+'))
    c.node('add2', nohtml(r'\+'))
    c.node('add3', nohtml(r'\+'))
    c.node('add4', nohtml(r'\+'))

with g.subgraph(name='in') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white')
    c.node('u_e_o', nohtml(r'user embedding\' | \[N, d\]'))
    c.node('i_e_o', nohtml(r'item embedding\' | \[M, d\]'))

g.edges([('u_e', 'spmm1:in'), ('i_e', 'spmm4:in'), ('sddmm:out', 'spmm2:in'), ('spmm2:out', 'Linear2:in')])
g.edge('sddmm:out', 'spmm3:in')

g.edge('weight1', 'Linear1:in')
g.edge('b1', 'Linear1:in')
g.edge('spmm2:out', 'Linear2:in')
g.edge('weight2', 'Linear2:in')
g.edge('b2', 'Linear2:in')
g.edge('spmm3:out', 'Linear3:in')
g.edge('weight1_1', 'Linear3:in')
g.edge('b1_1', 'Linear3:in')
g.edge('weight2_1', 'Linear4:in')
g.edge('b2_1', 'Linear4:in')

g.edges([('u_e', 'mul1'),  ('i_e', 'mul2')])
g.edge('norm_user', 'mul1', label='bc')
g.edge('norm_edge', 'spmm1:in', label='bc')
g.edge('norm_edge_1', 'spmm4:in', label='bc')
g.edge('norm_item', 'mul2', label='bc')

g.edge('mul1', 'sddmm:in')
g.edge('mul2', 'sddmm:in')
g.edge('Linear1:out', 'add1')
g.edge('Linear2:out', 'add1')

g.edge('Linear3:out', 'add2')
g.edge('Linear4:out', 'add2')
g.edges([('add1', 'i_e_o'), ('add2', 'u_e_o')])
g.edge('i_e_1', 'add3')
g.edge('spmm1', 'add3')
g.edge('add3', 'Linear1:in')
g.edge('u_e_1', 'add4')
g.edge('spmm4', 'add4')
g.edge('add4', 'Linear4:in')
g.render(filename='model5', directory="./",view=False)
import graphviz
from graphviz import nohtml

g = graphviz.Digraph('NGCF_model5', format='png', node_attr={'shape': 'record'}, engine='dot')
g.attr(rankdir='TD')
with g.subgraph(name='SpMM') as c:
    c.node_attr.update(color='gray', style='filled', fillcolor='aquamarine', shape='Mrecord')
    c.node('spmm1', nohtml(r"u_mul_e_sum\n| {<in> in| graph |<out> out}  | {two tensors | u\-\>i |\[M, d\]}"))
    c.node('spmm2', nohtml(r"copy_e_sum\n| {<in> in | graph | <out> out} | {one tensor | u\-\>i | \[M, d\]}"))
    c.node('spmm3', nohtml(r"copy_e_sum\n| {<in> in | graph | <out> out} | {one tensor | i\-\>u | \[N, d\]}"))
    c.node('spmm4', nohtml(r"u_mul_e_sum\n| {<in> in| graph | <out> out} | {two tensors | i\-\>u | \[N, d\]}"))


with g.subgraph(name='SDDMM') as c:
    c.node_attr.update(color='white', style='filled', fillcolor='darkturquoise', shape='Mrecord')
    c.node('sddmm', nohtml(r"u_mul_v\n| {<in> in | graph | <out> out} | {two tensors | u\-i | \[nnz, d\]}"))

with g.subgraph(name='Linear') as c:
    c.node_attr.update(color='white', style='filled', fillcolor='darkgreen', fontcolor='white', shape='Mrecord')
    c.node('Linear1', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[M, d\]}"))
    c.node('Linear2', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[M, d\]}"))
    c.node('Linear3', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[N, d\]}"))
    c.node('Linear4', nohtml(r"Linear\n| {<in> in | <out> out} | {three tensors | \[N, d\]}"))

with g.subgraph(name='norm') as c:
    c.node_attr.update(color='gray', style='filled', fillcolor='azure2')
    c.node('norm_user', nohtml(r"norm_user\n| \[N, 1\]"))
    c.node('norm_item', nohtml(r"norm_item\n| \[M, 1\]"))
    c.node('norm_edge', nohtml(r"norm_edge\n| \[nnz, 1\]"))
    c.node('norm_edge_1', nohtml(r"norm_edge\n| \[nnz, 1\]"))

with g.subgraph(name='in') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white')
    c.node('u_e', nohtml(r'user embedding | \[N, d\]'))
    c.node('i_e', nohtml(r'item embedding | \[M, d\]'))
    c.node('u_e_1', nohtml(r'user embedding | \[N, d\]'))
    c.node('i_e_1', nohtml(r'item embedding | \[M, d\]'))

with g.subgraph(name='weight') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='diamond')
    c.node('weight1', "W1")
    c.node('weight1_1', "W1")
    c.node('weight2', "W2")
    c.node('weight2_1', "W2")

with g.subgraph(name='bias') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='polygon')
    c.node('b1', "b1")
    c.node('b1_1', "b1")
    c.node('b2', "b2")
    c.node('b2_1', "b2")

with g.subgraph(name='operation') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white', shape='circle')
    c.node('mul1', nohtml(r'\*'))
    c.node('mul2', nohtml(r'\*'))
    c.node('add1', nohtml(r'\+'))
    c.node('add2', nohtml(r'\+'))
    c.node('add3', nohtml(r'\+'))
    c.node('add4', nohtml(r'\+'))

with g.subgraph(name='in') as c:
    c.node_attr.update(color='gray', stype='filled', fillcolor='white')
    c.node('u_e_o', nohtml(r'user embedding\' | \[N, d\]'))
    c.node('i_e_o', nohtml(r'item embedding\' | \[M, d\]'))

g.edges([('u_e', 'spmm1:in'), ('i_e', 'spmm4:in'), ('sddmm:out', 'spmm2:in'), ('spmm2:out', 'Linear2:in')])
g.edge('sddmm:out', 'spmm3:in')

g.edge('weight1', 'Linear1:in')
g.edge('b1', 'Linear1:in')
g.edge('spmm2:out', 'Linear2:in')
g.edge('weight2', 'Linear2:in')
g.edge('b2', 'Linear2:in')
g.edge('spmm3:out', 'Linear3:in')
g.edge('weight2_1', 'Linear3:in')
g.edge('b2_1', 'Linear3:in')
g.edge('weight1_1', 'Linear4:in')
g.edge('b1_1', 'Linear4:in')

g.edges([('u_e', 'mul1'),  ('i_e', 'mul2')])
g.edge('norm_user', 'mul1', label='bc')
g.edge('norm_edge', 'spmm1:in', label='bc')
g.edge('norm_edge_1', 'spmm4:in', label='bc')
g.edge('norm_item', 'mul2', label='bc')

g.edge('mul1', 'sddmm:in')
g.edge('mul2', 'sddmm:in')
g.edge('Linear1:out', 'add1')
g.edge('Linear2:out', 'add1')

g.edge('Linear3:out', 'add2')
g.edge('Linear4:out', 'add2')
g.edges([('add1', 'i_e_o'), ('add2', 'u_e_o')])
g.edge('i_e_1', 'add3')
g.edge('spmm1:out', 'add3')
g.edge('add3', 'Linear1:in')
g.edge('u_e_1', 'add4')
g.edge('spmm4:out', 'add4')
g.edge('add4', 'Linear4:in')
g.render(filename='model_5', directory="./",view=False)
