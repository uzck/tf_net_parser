чЕ
─%з%
:
Add
x"T
y"T
z"T"
Ttype:
2	
ю
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
Ы
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
8
Const
output"dtype"
valuetensor"
dtypetype
ы
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

С
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

Р
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
╘
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
ю
MaxPoolGrad

orig_input"T
orig_output"T	
grad"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype0:
2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
5

Reciprocal
x"T
y"T"
Ttype:

2	
D
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
А
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.7.02
b'unknown'б├
G
ConstConst*
value	B : *
dtype0*
_output_shapes
: 
l
Variable
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
shape: *
	container 
С
Variable/AssignAssignVariableConst*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable
a
Variable/readIdentityVariable*
T0*
_output_shapes
: *
_class
loc:@Variable
I
Const_1Const*
value	B : *
dtype0*
_output_shapes
: 
n

Variable_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
shape: *
	container 
Щ
Variable_1/AssignAssign
Variable_1Const_1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_1
g
Variable_1/readIdentity
Variable_1*
T0*
_output_shapes
: *
_class
loc:@Variable_1
x
inputPlaceholder*
dtype0*/
_output_shapes
:         *$
shape:         
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:         
*
shape:         

й
.conv2d/kernel/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0*
_output_shapes
:* 
_class
loc:@conv2d/kernel
У
,conv2d/kernel/Initializer/random_uniform/minConst*
valueB
 *nзо╜*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel
У
,conv2d/kernel/Initializer/random_uniform/maxConst*
valueB
 *nзо=*
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel
Ё
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*
seed2 * 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: *
dtype0*

seed 
╥
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: * 
_class
loc:@conv2d/kernel
ь
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
: * 
_class
loc:@conv2d/kernel
▐
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
: * 
_class
loc:@conv2d/kernel
│
conv2d/kernel
VariableV2*
shared_name * 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
dtype0*
	container *
shape: 
╙
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
А
conv2d/kernel/readIdentityconv2d/kernel*
T0*&
_output_shapes
: * 
_class
loc:@conv2d/kernel
Ч
-conv2d/bias/Initializer/zeros/shape_as_tensorConst*
valueB: *
dtype0*
_output_shapes
:*
_class
loc:@conv2d/bias
И
#conv2d/bias/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@conv2d/bias
╨
conv2d/bias/Initializer/zerosFill-conv2d/bias/Initializer/zeros/shape_as_tensor#conv2d/bias/Initializer/zeros/Const*
T0*
_output_shapes
: *

index_type0*
_class
loc:@conv2d/bias
Ч
conv2d/bias
VariableV2*
shared_name *
_class
loc:@conv2d/bias*
_output_shapes
: *
dtype0*
	container *
shape: 
╢
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
n
conv2d/bias/readIdentityconv2d/bias*
T0*
_output_shapes
: *
_class
loc:@conv2d/bias
e
conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
█
conv2d/Conv2DConv2Dinputconv2d/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:          *
use_cudnn_on_gpu(
Л
conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:          
]
conv2d/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:          
║
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:          
н
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel
Ч
.conv2d_1/kernel/Initializer/random_uniform/minConst*
valueB
 *═╠L╜*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
Ч
.conv2d_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *═╠L=*
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
Ў
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*
seed2 *"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: @*
dtype0*

seed 
┌
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
Ї
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*
T0*&
_output_shapes
: @*"
_class
loc:@conv2d_1/kernel
ц
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*
T0*&
_output_shapes
: @*"
_class
loc:@conv2d_1/kernel
╖
conv2d_1/kernel
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
dtype0*
	container *
shape: @
█
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
Ж
conv2d_1/kernel/readIdentityconv2d_1/kernel*
T0*&
_output_shapes
: @*"
_class
loc:@conv2d_1/kernel
Ы
/conv2d_1/bias/Initializer/zeros/shape_as_tensorConst*
valueB:@*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d_1/bias
М
%conv2d_1/bias/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_1/bias
╪
conv2d_1/bias/Initializer/zerosFill/conv2d_1/bias/Initializer/zeros/shape_as_tensor%conv2d_1/bias/Initializer/zeros/Const*
T0*
_output_shapes
:@*

index_type0* 
_class
loc:@conv2d_1/bias
Ы
conv2d_1/bias
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
╛
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
t
conv2d_1/bias/readIdentityconv2d_1/bias*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
g
conv2d_1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
я
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_1/kernel/read*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:         @*
use_cudnn_on_gpu(
С
conv2d_1/BiasAddBiasAddconv2d_1/Conv2Dconv2d_1/bias/read*
T0*
data_formatNHWC*/
_output_shapes
:         @
a
conv2d_1/ReluReluconv2d_1/BiasAdd*
T0*/
_output_shapes
:         @
╛
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:         @
^
Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
{
ReshapeReshapemax_pooling2d_1/MaxPoolReshape/shape*
T0*
Tshape0*(
_output_shapes
:         А
Я
-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel
С
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *╫│]╜*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel
С
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *╫│]=*
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel
ч
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
seed2 *
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
АА*
dtype0*

seed 
╬
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *
_class
loc:@dense/kernel
т
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
АА*
_class
loc:@dense/kernel
╘
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
АА*
_class
loc:@dense/kernel
е
dense/kernel
VariableV2*
shared_name *
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
dtype0*
	container *
shape:
АА
╔
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
w
dense/kernel/readIdentitydense/kernel*
T0* 
_output_shapes
:
АА*
_class
loc:@dense/kernel
Ц
,dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:А*
dtype0*
_output_shapes
:*
_class
loc:@dense/bias
Ж
"dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@dense/bias
═
dense/bias/Initializer/zerosFill,dense/bias/Initializer/zeros/shape_as_tensor"dense/bias/Initializer/zeros/Const*
T0*
_output_shapes	
:А*

index_type0*
_class
loc:@dense/bias
Ч

dense/bias
VariableV2*
shared_name *
_class
loc:@dense/bias*
_output_shapes	
:А*
dtype0*
	container *
shape:А
│
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
l
dense/bias/readIdentity
dense/bias*
T0*
_output_shapes	
:А*
_class
loc:@dense/bias
Л
dense/MatMulMatMulReshapedense/kernel/read*
transpose_b( *
T0*
transpose_a( *(
_output_shapes
:         А
Б
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:         А
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:         А
g
truncated_normal/shapeConst*
valueB"   
   *
dtype0*
_output_shapes
:
Z
truncated_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
truncated_normal/stddevConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Ы
 truncated_normal/TruncatedNormalTruncatedNormaltruncated_normal/shape*
T0*
dtype0*
_output_shapes
:	А
*
seed2 *

seed 
А
truncated_normal/mulMul truncated_normal/TruncatedNormaltruncated_normal/stddev*
T0*
_output_shapes
:	А

n
truncated_normalAddtruncated_normal/multruncated_normal/mean*
T0*
_output_shapes
:	А

А

Variable_2
VariableV2*
dtype0*
_output_shapes
:	А
*
shared_name *
shape:	А
*
	container 
л
Variable_2/AssignAssign
Variable_2truncated_normal*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
p
Variable_2/readIdentity
Variable_2*
T0*
_output_shapes
:	А
*
_class
loc:@Variable_2
Е
MatMulMatMul
dense/ReluVariable_2/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:         

L
SoftmaxSoftmaxMatMul*
T0*'
_output_shapes
:         

U
predict-resultIdentitySoftmax*
T0*'
_output_shapes
:         

R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
ArgMaxArgMaxSoftmaxArgMax/dimension*
output_type0	*
T0*#
_output_shapes
:         *

Tidx0
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
А
ArgMax_1ArgMaxPlaceholderArgMax_1/dimension*
output_type0	*
T0*#
_output_shapes
:         *

Tidx0
N
EqualEqualArgMaxArgMax_1*
T0	*#
_output_shapes
:         
P
CastCastEqual*

DstT0*#
_output_shapes
:         *

SrcT0

Q
Const_2Const*
valueB: *
dtype0*
_output_shapes
:
Y
MeanMeanCastConst_2*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
E
LogLogSoftmax*
T0*'
_output_shapes
:         

N
mulMulPlaceholderLog*
T0*'
_output_shapes
:         

X
Const_3Const*
valueB"       *
dtype0*
_output_shapes
:
V
SumSummulConst_3*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
0
NegNegSum*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
q
 gradients/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ц
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
[
gradients/Sum_grad/ShapeShapemul*
T0*
out_type0*
_output_shapes
:
Щ
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*
T0*'
_output_shapes
:         
*

Tmultiples0
c
gradients/mul_grad/ShapeShapePlaceholder*
T0*
out_type0*
_output_shapes
:
]
gradients/mul_grad/Shape_1ShapeLog*
T0*
out_type0*
_output_shapes
:
┤
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
T0*2
_output_shapes 
:         :         
m
gradients/mul_grad/MulMulgradients/Sum_grad/TileLog*
T0*'
_output_shapes
:         

Я
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Ч
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         

w
gradients/mul_grad/Mul_1MulPlaceholdergradients/Sum_grad/Tile*
T0*'
_output_shapes
:         

е
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
Э
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         

g
#gradients/mul_grad/tuple/group_depsNoOp^gradients/mul_grad/Reshape^gradients/mul_grad/Reshape_1
┌
+gradients/mul_grad/tuple/control_dependencyIdentitygradients/mul_grad/Reshape$^gradients/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:         
*-
_class#
!loc:@gradients/mul_grad/Reshape
р
-gradients/mul_grad/tuple/control_dependency_1Identitygradients/mul_grad/Reshape_1$^gradients/mul_grad/tuple/group_deps*
T0*'
_output_shapes
:         
*/
_class%
#!loc:@gradients/mul_grad/Reshape_1
Ц
gradients/Log_grad/Reciprocal
ReciprocalSoftmax.^gradients/mul_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:         

Э
gradients/Log_grad/mulMul-gradients/mul_grad/tuple/control_dependency_1gradients/Log_grad/Reciprocal*
T0*'
_output_shapes
:         

t
gradients/Softmax_grad/mulMulgradients/Log_grad/mulSoftmax*
T0*'
_output_shapes
:         

v
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
╢
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*
	keep_dims( *
T0*#
_output_shapes
:         *

Tidx0
u
$gradients/Softmax_grad/Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:
л
gradients/Softmax_grad/ReshapeReshapegradients/Softmax_grad/Sum$gradients/Softmax_grad/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:         
Л
gradients/Softmax_grad/subSubgradients/Log_grad/mulgradients/Softmax_grad/Reshape*
T0*'
_output_shapes
:         

z
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0*'
_output_shapes
:         

о
gradients/MatMul_grad/MatMulMatMulgradients/Softmax_grad/mul_1Variable_2/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:         А
в
gradients/MatMul_grad/MatMul_1MatMul
dense/Relugradients/Softmax_grad/mul_1*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	А

n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
х
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:         А*/
_class%
#!loc:@gradients/MatMul_grad/MatMul
т
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	А
*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1
Э
"gradients/dense/Relu_grad/ReluGradReluGrad.gradients/MatMul_grad/tuple/control_dependency
dense/Relu*
T0*(
_output_shapes
:         А
Ш
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
Е
-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp#^gradients/dense/Relu_grad/ReluGrad)^gradients/dense/BiasAdd_grad/BiasAddGrad
 
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentity"gradients/dense/Relu_grad/ReluGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*(
_output_shapes
:         А*5
_class+
)'loc:@gradients/dense/Relu_grad/ReluGrad
А
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:А*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad
╧
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:         А
┐
$gradients/dense/MatMul_grad/MatMul_1MatMulReshape5gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:
АА
А
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
¤
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*(
_output_shapes
:         А*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul
√
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
АА*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1
s
gradients/Reshape_grad/ShapeShapemax_pooling2d_1/MaxPool*
T0*
out_type0*
_output_shapes
:
┼
gradients/Reshape_grad/ReshapeReshape4gradients/dense/MatMul_grad/tuple/control_dependencygradients/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:         @
Ц
2gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d_1/Relumax_pooling2d_1/MaxPoolgradients/Reshape_grad/Reshape*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:         @
о
%gradients/conv2d_1/Relu_grad/ReluGradReluGrad2gradients/max_pooling2d_1/MaxPool_grad/MaxPoolGradconv2d_1/Relu*
T0*/
_output_shapes
:         @
Э
+gradients/conv2d_1/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/conv2d_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
О
0gradients/conv2d_1/BiasAdd_grad/tuple/group_depsNoOp&^gradients/conv2d_1/Relu_grad/ReluGrad,^gradients/conv2d_1/BiasAdd_grad/BiasAddGrad
Т
8gradients/conv2d_1/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/conv2d_1/Relu_grad/ReluGrad1^gradients/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:         @*8
_class.
,*loc:@gradients/conv2d_1/Relu_grad/ReluGrad
Л
:gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/conv2d_1/BiasAdd_grad/BiasAddGrad1^gradients/conv2d_1/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:@*>
_class4
20loc:@gradients/conv2d_1/BiasAdd_grad/BiasAddGrad
а
%gradients/conv2d_1/Conv2D_grad/ShapeNShapeNmax_pooling2d/MaxPoolconv2d_1/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
}
$gradients/conv2d_1/Conv2D_grad/ConstConst*%
valueB"          @   *
dtype0*
_output_shapes
:
Д
2gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput%gradients/conv2d_1/Conv2D_grad/ShapeNconv2d_1/kernel/read8gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4                                    *
use_cudnn_on_gpu(
т
3gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltermax_pooling2d/MaxPool$gradients/conv2d_1/Conv2D_grad/Const8gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*&
_output_shapes
: @*
use_cudnn_on_gpu(
в
/gradients/conv2d_1/Conv2D_grad/tuple/group_depsNoOp3^gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput4^gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter
к
7gradients/conv2d_1/Conv2D_grad/tuple/control_dependencyIdentity2gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput0^gradients/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:          *E
_class;
97loc:@gradients/conv2d_1/Conv2D_grad/Conv2DBackpropInput
е
9gradients/conv2d_1/Conv2D_grad/tuple/control_dependency_1Identity3gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter0^gradients/conv2d_1/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
: @*F
_class<
:8loc:@gradients/conv2d_1/Conv2D_grad/Conv2DBackpropFilter
й
0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradMaxPoolGradconv2d/Relumax_pooling2d/MaxPool7gradients/conv2d_1/Conv2D_grad/tuple/control_dependency*
strides
*
ksize
*
T0*
data_formatNHWC*
paddingVALID*/
_output_shapes
:          
и
#gradients/conv2d/Relu_grad/ReluGradReluGrad0gradients/max_pooling2d/MaxPool_grad/MaxPoolGradconv2d/Relu*
T0*/
_output_shapes
:          
Щ
)gradients/conv2d/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients/conv2d/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
И
.gradients/conv2d/BiasAdd_grad/tuple/group_depsNoOp$^gradients/conv2d/Relu_grad/ReluGrad*^gradients/conv2d/BiasAdd_grad/BiasAddGrad
К
6gradients/conv2d/BiasAdd_grad/tuple/control_dependencyIdentity#gradients/conv2d/Relu_grad/ReluGrad/^gradients/conv2d/BiasAdd_grad/tuple/group_deps*
T0*/
_output_shapes
:          *6
_class,
*(loc:@gradients/conv2d/Relu_grad/ReluGrad
Г
8gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1Identity)gradients/conv2d/BiasAdd_grad/BiasAddGrad/^gradients/conv2d/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
: *<
_class2
0.loc:@gradients/conv2d/BiasAdd_grad/BiasAddGrad
М
#gradients/conv2d/Conv2D_grad/ShapeNShapeNinputconv2d/kernel/read*
T0*
out_type0*
N* 
_output_shapes
::
{
"gradients/conv2d/Conv2D_grad/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
№
0gradients/conv2d/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput#gradients/conv2d/Conv2D_grad/ShapeNconv2d/kernel/read6gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*J
_output_shapes8
6:4                                    *
use_cudnn_on_gpu(
╠
1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterinput"gradients/conv2d/Conv2D_grad/Const6gradients/conv2d/BiasAdd_grad/tuple/control_dependency*
strides
*
	dilations
*
T0*
data_formatNHWC*
paddingVALID*&
_output_shapes
: *
use_cudnn_on_gpu(
Ь
-gradients/conv2d/Conv2D_grad/tuple/group_depsNoOp1^gradients/conv2d/Conv2D_grad/Conv2DBackpropInput2^gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
в
5gradients/conv2d/Conv2D_grad/tuple/control_dependencyIdentity0gradients/conv2d/Conv2D_grad/Conv2DBackpropInput.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*/
_output_shapes
:         *C
_class9
75loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropInput
Э
7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1Identity1gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter.^gradients/conv2d/Conv2D_grad/tuple/group_deps*
T0*&
_output_shapes
: *D
_class:
86loc:@gradients/conv2d/Conv2D_grad/Conv2DBackpropFilter
}
beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*
_output_shapes
: *
_class
loc:@Variable_2
О
beta1_power
VariableV2*
shared_name *
_class
loc:@Variable_2*
_output_shapes
: *
dtype0*
	container *
shape: 
н
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
i
beta1_power/readIdentitybeta1_power*
T0*
_output_shapes
: *
_class
loc:@Variable_2
}
beta2_power/initial_valueConst*
valueB
 *w╛?*
dtype0*
_output_shapes
: *
_class
loc:@Variable_2
О
beta2_power
VariableV2*
shared_name *
_class
loc:@Variable_2*
_output_shapes
: *
dtype0*
	container *
shape: 
н
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
i
beta2_power/readIdentitybeta2_power*
T0*
_output_shapes
: *
_class
loc:@Variable_2
п
4conv2d/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"             *
dtype0*
_output_shapes
:* 
_class
loc:@conv2d/kernel
С
*conv2d/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel
є
$conv2d/kernel/Adam/Initializer/zerosFill4conv2d/kernel/Adam/Initializer/zeros/shape_as_tensor*conv2d/kernel/Adam/Initializer/zeros/Const*
T0*&
_output_shapes
: *

index_type0* 
_class
loc:@conv2d/kernel
╕
conv2d/kernel/Adam
VariableV2*
shared_name * 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
dtype0*
	container *
shape: 
┘
conv2d/kernel/Adam/AssignAssignconv2d/kernel/Adam$conv2d/kernel/Adam/Initializer/zeros*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
К
conv2d/kernel/Adam/readIdentityconv2d/kernel/Adam*
T0*&
_output_shapes
: * 
_class
loc:@conv2d/kernel
▒
6conv2d/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"             *
dtype0*
_output_shapes
:* 
_class
loc:@conv2d/kernel
У
,conv2d/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: * 
_class
loc:@conv2d/kernel
∙
&conv2d/kernel/Adam_1/Initializer/zerosFill6conv2d/kernel/Adam_1/Initializer/zeros/shape_as_tensor,conv2d/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_output_shapes
: *

index_type0* 
_class
loc:@conv2d/kernel
║
conv2d/kernel/Adam_1
VariableV2*
shared_name * 
_class
loc:@conv2d/kernel*&
_output_shapes
: *
dtype0*
	container *
shape: 
▀
conv2d/kernel/Adam_1/AssignAssignconv2d/kernel/Adam_1&conv2d/kernel/Adam_1/Initializer/zeros*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
О
conv2d/kernel/Adam_1/readIdentityconv2d/kernel/Adam_1*
T0*&
_output_shapes
: * 
_class
loc:@conv2d/kernel
Ь
2conv2d/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB: *
dtype0*
_output_shapes
:*
_class
loc:@conv2d/bias
Н
(conv2d/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@conv2d/bias
▀
"conv2d/bias/Adam/Initializer/zerosFill2conv2d/bias/Adam/Initializer/zeros/shape_as_tensor(conv2d/bias/Adam/Initializer/zeros/Const*
T0*
_output_shapes
: *

index_type0*
_class
loc:@conv2d/bias
Ь
conv2d/bias/Adam
VariableV2*
shared_name *
_class
loc:@conv2d/bias*
_output_shapes
: *
dtype0*
	container *
shape: 
┼
conv2d/bias/Adam/AssignAssignconv2d/bias/Adam"conv2d/bias/Adam/Initializer/zeros*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
x
conv2d/bias/Adam/readIdentityconv2d/bias/Adam*
T0*
_output_shapes
: *
_class
loc:@conv2d/bias
Ю
4conv2d/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB: *
dtype0*
_output_shapes
:*
_class
loc:@conv2d/bias
П
*conv2d/bias/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@conv2d/bias
х
$conv2d/bias/Adam_1/Initializer/zerosFill4conv2d/bias/Adam_1/Initializer/zeros/shape_as_tensor*conv2d/bias/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
: *

index_type0*
_class
loc:@conv2d/bias
Ю
conv2d/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@conv2d/bias*
_output_shapes
: *
dtype0*
	container *
shape: 
╦
conv2d/bias/Adam_1/AssignAssignconv2d/bias/Adam_1$conv2d/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
|
conv2d/bias/Adam_1/readIdentityconv2d/bias/Adam_1*
T0*
_output_shapes
: *
_class
loc:@conv2d/bias
│
6conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel
Х
,conv2d_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
√
&conv2d_1/kernel/Adam/Initializer/zerosFill6conv2d_1/kernel/Adam/Initializer/zeros/shape_as_tensor,conv2d_1/kernel/Adam/Initializer/zeros/Const*
T0*&
_output_shapes
: @*

index_type0*"
_class
loc:@conv2d_1/kernel
╝
conv2d_1/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
dtype0*
	container *
shape: @
с
conv2d_1/kernel/Adam/AssignAssignconv2d_1/kernel/Adam&conv2d_1/kernel/Adam/Initializer/zeros*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
Р
conv2d_1/kernel/Adam/readIdentityconv2d_1/kernel/Adam*
T0*&
_output_shapes
: @*"
_class
loc:@conv2d_1/kernel
╡
8conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *
dtype0*
_output_shapes
:*"
_class
loc:@conv2d_1/kernel
Ч
.conv2d_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *"
_class
loc:@conv2d_1/kernel
Б
(conv2d_1/kernel/Adam_1/Initializer/zerosFill8conv2d_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor.conv2d_1/kernel/Adam_1/Initializer/zeros/Const*
T0*&
_output_shapes
: @*

index_type0*"
_class
loc:@conv2d_1/kernel
╛
conv2d_1/kernel/Adam_1
VariableV2*
shared_name *"
_class
loc:@conv2d_1/kernel*&
_output_shapes
: @*
dtype0*
	container *
shape: @
ч
conv2d_1/kernel/Adam_1/AssignAssignconv2d_1/kernel/Adam_1(conv2d_1/kernel/Adam_1/Initializer/zeros*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
Ф
conv2d_1/kernel/Adam_1/readIdentityconv2d_1/kernel/Adam_1*
T0*&
_output_shapes
: @*"
_class
loc:@conv2d_1/kernel
а
4conv2d_1/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:@*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d_1/bias
С
*conv2d_1/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_1/bias
ч
$conv2d_1/bias/Adam/Initializer/zerosFill4conv2d_1/bias/Adam/Initializer/zeros/shape_as_tensor*conv2d_1/bias/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:@*

index_type0* 
_class
loc:@conv2d_1/bias
а
conv2d_1/bias/Adam
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
═
conv2d_1/bias/Adam/AssignAssignconv2d_1/bias/Adam$conv2d_1/bias/Adam/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
~
conv2d_1/bias/Adam/readIdentityconv2d_1/bias/Adam*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
в
6conv2d_1/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:@*
dtype0*
_output_shapes
:* 
_class
loc:@conv2d_1/bias
У
,conv2d_1/bias/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: * 
_class
loc:@conv2d_1/bias
э
&conv2d_1/bias/Adam_1/Initializer/zerosFill6conv2d_1/bias/Adam_1/Initializer/zeros/shape_as_tensor,conv2d_1/bias/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
:@*

index_type0* 
_class
loc:@conv2d_1/bias
в
conv2d_1/bias/Adam_1
VariableV2*
shared_name * 
_class
loc:@conv2d_1/bias*
_output_shapes
:@*
dtype0*
	container *
shape:@
╙
conv2d_1/bias/Adam_1/AssignAssignconv2d_1/bias/Adam_1&conv2d_1/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
В
conv2d_1/bias/Adam_1/readIdentityconv2d_1/bias/Adam_1*
T0*
_output_shapes
:@* 
_class
loc:@conv2d_1/bias
е
3dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel
П
)dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel
щ
#dense/kernel/Adam/Initializer/zerosFill3dense/kernel/Adam/Initializer/zeros/shape_as_tensor)dense/kernel/Adam/Initializer/zeros/Const*
T0* 
_output_shapes
:
АА*

index_type0*
_class
loc:@dense/kernel
к
dense/kernel/Adam
VariableV2*
shared_name *
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
dtype0*
	container *
shape:
АА
╧
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
Б
dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0* 
_output_shapes
:
АА*
_class
loc:@dense/kernel
з
5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *
dtype0*
_output_shapes
:*
_class
loc:@dense/kernel
С
+dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@dense/kernel
я
%dense/kernel/Adam_1/Initializer/zerosFill5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor+dense/kernel/Adam_1/Initializer/zeros/Const*
T0* 
_output_shapes
:
АА*

index_type0*
_class
loc:@dense/kernel
м
dense/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@dense/kernel* 
_output_shapes
:
АА*
dtype0*
	container *
shape:
АА
╒
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
Е
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0* 
_output_shapes
:
АА*
_class
loc:@dense/kernel
Ы
1dense/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:А*
dtype0*
_output_shapes
:*
_class
loc:@dense/bias
Л
'dense/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@dense/bias
▄
!dense/bias/Adam/Initializer/zerosFill1dense/bias/Adam/Initializer/zeros/shape_as_tensor'dense/bias/Adam/Initializer/zeros/Const*
T0*
_output_shapes	
:А*

index_type0*
_class
loc:@dense/bias
Ь
dense/bias/Adam
VariableV2*
shared_name *
_class
loc:@dense/bias*
_output_shapes	
:А*
dtype0*
	container *
shape:А
┬
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
v
dense/bias/Adam/readIdentitydense/bias/Adam*
T0*
_output_shapes	
:А*
_class
loc:@dense/bias
Э
3dense/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:А*
dtype0*
_output_shapes
:*
_class
loc:@dense/bias
Н
)dense/bias/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@dense/bias
т
#dense/bias/Adam_1/Initializer/zerosFill3dense/bias/Adam_1/Initializer/zeros/shape_as_tensor)dense/bias/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes	
:А*

index_type0*
_class
loc:@dense/bias
Ю
dense/bias/Adam_1
VariableV2*
shared_name *
_class
loc:@dense/bias*
_output_shapes	
:А*
dtype0*
	container *
shape:А
╚
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
z
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_output_shapes	
:А*
_class
loc:@dense/bias
б
1Variable_2/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *
dtype0*
_output_shapes
:*
_class
loc:@Variable_2
Л
'Variable_2/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@Variable_2
р
!Variable_2/Adam/Initializer/zerosFill1Variable_2/Adam/Initializer/zeros/shape_as_tensor'Variable_2/Adam/Initializer/zeros/Const*
T0*
_output_shapes
:	А
*

index_type0*
_class
loc:@Variable_2
д
Variable_2/Adam
VariableV2*
shared_name *
_class
loc:@Variable_2*
_output_shapes
:	А
*
dtype0*
	container *
shape:	А

╞
Variable_2/Adam/AssignAssignVariable_2/Adam!Variable_2/Adam/Initializer/zeros*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
z
Variable_2/Adam/readIdentityVariable_2/Adam*
T0*
_output_shapes
:	А
*
_class
loc:@Variable_2
г
3Variable_2/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *
dtype0*
_output_shapes
:*
_class
loc:@Variable_2
Н
)Variable_2/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: *
_class
loc:@Variable_2
ц
#Variable_2/Adam_1/Initializer/zerosFill3Variable_2/Adam_1/Initializer/zeros/shape_as_tensor)Variable_2/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
:	А
*

index_type0*
_class
loc:@Variable_2
ж
Variable_2/Adam_1
VariableV2*
shared_name *
_class
loc:@Variable_2*
_output_shapes
:	А
*
dtype0*
	container *
shape:	А

╠
Variable_2/Adam_1/AssignAssignVariable_2/Adam_1#Variable_2/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
~
Variable_2/Adam_1/readIdentityVariable_2/Adam_1*
T0*
_output_shapes
:	А
*
_class
loc:@Variable_2
W
Adam/learning_rateConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w╛?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
·
#Adam/update_conv2d/kernel/ApplyAdam	ApplyAdamconv2d/kernelconv2d/kernel/Adamconv2d/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/conv2d/Conv2D_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: *
use_locking( * 
_class
loc:@conv2d/kernel*
use_nesterov( 
х
!Adam/update_conv2d/bias/ApplyAdam	ApplyAdamconv2d/biasconv2d/bias/Adamconv2d/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon8gradients/conv2d/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
: *
use_locking( *
_class
loc:@conv2d/bias*
use_nesterov( 
Ж
%Adam/update_conv2d_1/kernel/ApplyAdam	ApplyAdamconv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon9gradients/conv2d_1/Conv2D_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: @*
use_locking( *"
_class
loc:@conv2d_1/kernel*
use_nesterov( 
ё
#Adam/update_conv2d_1/bias/ApplyAdam	ApplyAdamconv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon:gradients/conv2d_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:@*
use_locking( * 
_class
loc:@conv2d_1/bias*
use_nesterov( 
ю
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
АА*
use_locking( *
_class
loc:@dense/kernel*
use_nesterov( 
р
 Adam/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А*
use_locking( *
_class
loc:@dense/bias*
use_nesterov( 
▌
 Adam/update_Variable_2/ApplyAdam	ApplyAdam
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	А
*
use_locking( *
_class
loc:@Variable_2*
use_nesterov( 
Ё
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_conv2d/kernel/ApplyAdam"^Adam/update_conv2d/bias/ApplyAdam&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam!^Adam/update_Variable_2/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable_2
Х
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*
_class
loc:@Variable_2
Є

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_conv2d/kernel/ApplyAdam"^Adam/update_conv2d/bias/ApplyAdam&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam!^Adam/update_Variable_2/ApplyAdam*
T0*
_output_shapes
: *
_class
loc:@Variable_2
Щ
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_output_shapes
: *
use_locking( *
validate_shape(*
_class
loc:@Variable_2
н
AdamNoOp$^Adam/update_conv2d/kernel/ApplyAdam"^Adam/update_conv2d/bias/ApplyAdam&^Adam/update_conv2d_1/kernel/ApplyAdam$^Adam/update_conv2d_1/bias/ApplyAdam#^Adam/update_dense/kernel/ApplyAdam!^Adam/update_dense/bias/ApplyAdam!^Adam/update_Variable_2/ApplyAdam^Adam/Assign^Adam/Assign_1
■
initNoOp^Variable/Assign^Variable_1/Assign^conv2d/kernel/Assign^conv2d/bias/Assign^conv2d_1/kernel/Assign^conv2d_1/bias/Assign^dense/kernel/Assign^dense/bias/Assign^Variable_2/Assign^beta1_power/Assign^beta2_power/Assign^conv2d/kernel/Adam/Assign^conv2d/kernel/Adam_1/Assign^conv2d/bias/Adam/Assign^conv2d/bias/Adam_1/Assign^conv2d_1/kernel/Adam/Assign^conv2d_1/kernel/Adam_1/Assign^conv2d_1/bias/Adam/Assign^conv2d_1/bias/Adam_1/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^Variable_2/Adam/Assign^Variable_2/Adam_1/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_ae350235bd0645849cac57e3be821975/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
М
save/SaveV2/tensor_namesConst*┐
value╡B▓BVariableB
Variable_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1Bbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
Х
save/SaveV2/shape_and_slicesConst*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
╢
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_powerbeta2_powerconv2d/biasconv2d/bias/Adamconv2d/bias/Adam_1conv2d/kernelconv2d/kernel/Adamconv2d/kernel/Adam_1conv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1conv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1*'
dtypes
2
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints*
T0*
_output_shapes
: 
П
save/RestoreV2/tensor_namesConst*┐
value╡B▓BVariableB
Variable_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1Bbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
Ш
save/RestoreV2/shape_and_slicesConst*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
И
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*'
dtypes
2*x
_output_shapesf
d:::::::::::::::::::::::::
Ц
save/AssignAssignVariablesave/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable
Ю
save/Assign_1Assign
Variable_1save/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_1
з
save/Assign_2Assign
Variable_2save/RestoreV2:2*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
м
save/Assign_3AssignVariable_2/Adamsave/RestoreV2:3*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
о
save/Assign_4AssignVariable_2/Adam_1save/RestoreV2:4*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
Я
save/Assign_5Assignbeta1_powersave/RestoreV2:5*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
Я
save/Assign_6Assignbeta2_powersave/RestoreV2:6*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
д
save/Assign_7Assignconv2d/biassave/RestoreV2:7*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
й
save/Assign_8Assignconv2d/bias/Adamsave/RestoreV2:8*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
л
save/Assign_9Assignconv2d/bias/Adam_1save/RestoreV2:9*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
╢
save/Assign_10Assignconv2d/kernelsave/RestoreV2:10*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
╗
save/Assign_11Assignconv2d/kernel/Adamsave/RestoreV2:11*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
╜
save/Assign_12Assignconv2d/kernel/Adam_1save/RestoreV2:12*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
к
save/Assign_13Assignconv2d_1/biassave/RestoreV2:13*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
п
save/Assign_14Assignconv2d_1/bias/Adamsave/RestoreV2:14*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
▒
save/Assign_15Assignconv2d_1/bias/Adam_1save/RestoreV2:15*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
║
save/Assign_16Assignconv2d_1/kernelsave/RestoreV2:16*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
┐
save/Assign_17Assignconv2d_1/kernel/Adamsave/RestoreV2:17*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
┴
save/Assign_18Assignconv2d_1/kernel/Adam_1save/RestoreV2:18*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
е
save/Assign_19Assign
dense/biassave/RestoreV2:19*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
к
save/Assign_20Assigndense/bias/Adamsave/RestoreV2:20*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
м
save/Assign_21Assigndense/bias/Adam_1save/RestoreV2:21*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
о
save/Assign_22Assigndense/kernelsave/RestoreV2:22*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
│
save/Assign_23Assigndense/kernel/Adamsave/RestoreV2:23*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
╡
save/Assign_24Assigndense/kernel/Adam_1save/RestoreV2:24*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
╖
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24
-
save/restore_allNoOp^save/restore_shard
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ж
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_adf5ad787ea74609be7c55ecfeef9247/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
О
save_1/SaveV2/tensor_namesConst*┐
value╡B▓BVariableB
Variable_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1Bbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
Ч
save_1/SaveV2/shape_and_slicesConst*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
╛
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_powerbeta2_powerconv2d/biasconv2d/bias/Adamconv2d/bias/Adam_1conv2d/kernelconv2d/kernel/Adamconv2d/kernel/Adam_1conv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1conv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1*'
dtypes
2
Щ
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
г
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
T0*
N*
_output_shapes
:
Г
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
В
save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints*
T0*
_output_shapes
: 
С
save_1/RestoreV2/tensor_namesConst*┐
value╡B▓BVariableB
Variable_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1Bbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
Ъ
!save_1/RestoreV2/shape_and_slicesConst*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Р
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*'
dtypes
2*x
_output_shapesf
d:::::::::::::::::::::::::
Ъ
save_1/AssignAssignVariablesave_1/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable
в
save_1/Assign_1Assign
Variable_1save_1/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_1
л
save_1/Assign_2Assign
Variable_2save_1/RestoreV2:2*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
░
save_1/Assign_3AssignVariable_2/Adamsave_1/RestoreV2:3*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
▓
save_1/Assign_4AssignVariable_2/Adam_1save_1/RestoreV2:4*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
г
save_1/Assign_5Assignbeta1_powersave_1/RestoreV2:5*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
г
save_1/Assign_6Assignbeta2_powersave_1/RestoreV2:6*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
и
save_1/Assign_7Assignconv2d/biassave_1/RestoreV2:7*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
н
save_1/Assign_8Assignconv2d/bias/Adamsave_1/RestoreV2:8*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
п
save_1/Assign_9Assignconv2d/bias/Adam_1save_1/RestoreV2:9*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
║
save_1/Assign_10Assignconv2d/kernelsave_1/RestoreV2:10*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
┐
save_1/Assign_11Assignconv2d/kernel/Adamsave_1/RestoreV2:11*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
┴
save_1/Assign_12Assignconv2d/kernel/Adam_1save_1/RestoreV2:12*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
о
save_1/Assign_13Assignconv2d_1/biassave_1/RestoreV2:13*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
│
save_1/Assign_14Assignconv2d_1/bias/Adamsave_1/RestoreV2:14*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
╡
save_1/Assign_15Assignconv2d_1/bias/Adam_1save_1/RestoreV2:15*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
╛
save_1/Assign_16Assignconv2d_1/kernelsave_1/RestoreV2:16*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
├
save_1/Assign_17Assignconv2d_1/kernel/Adamsave_1/RestoreV2:17*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
┼
save_1/Assign_18Assignconv2d_1/kernel/Adam_1save_1/RestoreV2:18*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
й
save_1/Assign_19Assign
dense/biassave_1/RestoreV2:19*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
о
save_1/Assign_20Assigndense/bias/Adamsave_1/RestoreV2:20*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
░
save_1/Assign_21Assigndense/bias/Adam_1save_1/RestoreV2:21*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
▓
save_1/Assign_22Assigndense/kernelsave_1/RestoreV2:22*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
╖
save_1/Assign_23Assigndense/kernel/Adamsave_1/RestoreV2:23*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
╣
save_1/Assign_24Assigndense/kernel/Adam_1save_1/RestoreV2:24*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
ы
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24
1
save_1/restore_allNoOp^save_1/restore_shard
R
save_2/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ж
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_78d397c60d754c6b921d84c3d2ef0354/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Е
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
О
save_2/SaveV2/tensor_namesConst*┐
value╡B▓BVariableB
Variable_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1Bbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
Ч
save_2/SaveV2/shape_and_slicesConst*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
╛
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesVariable
Variable_1
Variable_2Variable_2/AdamVariable_2/Adam_1beta1_powerbeta2_powerconv2d/biasconv2d/bias/Adamconv2d/bias/Adam_1conv2d/kernelconv2d/kernel/Adamconv2d/kernel/Adam_1conv2d_1/biasconv2d_1/bias/Adamconv2d_1/bias/Adam_1conv2d_1/kernelconv2d_1/kernel/Adamconv2d_1/kernel/Adam_1
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1*'
dtypes
2
Щ
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_2/ShardedFilename
г
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*

axis *
T0*
N*
_output_shapes
:
Г
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
В
save_2/IdentityIdentitysave_2/Const^save_2/control_dependency^save_2/MergeV2Checkpoints*
T0*
_output_shapes
: 
С
save_2/RestoreV2/tensor_namesConst*┐
value╡B▓BVariableB
Variable_1B
Variable_2BVariable_2/AdamBVariable_2/Adam_1Bbeta1_powerBbeta2_powerBconv2d/biasBconv2d/bias/AdamBconv2d/bias/Adam_1Bconv2d/kernelBconv2d/kernel/AdamBconv2d/kernel/Adam_1Bconv2d_1/biasBconv2d_1/bias/AdamBconv2d_1/bias/Adam_1Bconv2d_1/kernelBconv2d_1/kernel/AdamBconv2d_1/kernel/Adam_1B
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1*
dtype0*
_output_shapes
:
Ъ
!save_2/RestoreV2/shape_and_slicesConst*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Р
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*'
dtypes
2*x
_output_shapesf
d:::::::::::::::::::::::::
Ъ
save_2/AssignAssignVariablesave_2/RestoreV2*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable
в
save_2/Assign_1Assign
Variable_1save_2/RestoreV2:1*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_1
л
save_2/Assign_2Assign
Variable_2save_2/RestoreV2:2*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
░
save_2/Assign_3AssignVariable_2/Adamsave_2/RestoreV2:3*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
▓
save_2/Assign_4AssignVariable_2/Adam_1save_2/RestoreV2:4*
T0*
_output_shapes
:	А
*
use_locking(*
validate_shape(*
_class
loc:@Variable_2
г
save_2/Assign_5Assignbeta1_powersave_2/RestoreV2:5*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
г
save_2/Assign_6Assignbeta2_powersave_2/RestoreV2:6*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@Variable_2
и
save_2/Assign_7Assignconv2d/biassave_2/RestoreV2:7*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
н
save_2/Assign_8Assignconv2d/bias/Adamsave_2/RestoreV2:8*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
п
save_2/Assign_9Assignconv2d/bias/Adam_1save_2/RestoreV2:9*
T0*
_output_shapes
: *
use_locking(*
validate_shape(*
_class
loc:@conv2d/bias
║
save_2/Assign_10Assignconv2d/kernelsave_2/RestoreV2:10*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
┐
save_2/Assign_11Assignconv2d/kernel/Adamsave_2/RestoreV2:11*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
┴
save_2/Assign_12Assignconv2d/kernel/Adam_1save_2/RestoreV2:12*
T0*&
_output_shapes
: *
use_locking(*
validate_shape(* 
_class
loc:@conv2d/kernel
о
save_2/Assign_13Assignconv2d_1/biassave_2/RestoreV2:13*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
│
save_2/Assign_14Assignconv2d_1/bias/Adamsave_2/RestoreV2:14*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
╡
save_2/Assign_15Assignconv2d_1/bias/Adam_1save_2/RestoreV2:15*
T0*
_output_shapes
:@*
use_locking(*
validate_shape(* 
_class
loc:@conv2d_1/bias
╛
save_2/Assign_16Assignconv2d_1/kernelsave_2/RestoreV2:16*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
├
save_2/Assign_17Assignconv2d_1/kernel/Adamsave_2/RestoreV2:17*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
┼
save_2/Assign_18Assignconv2d_1/kernel/Adam_1save_2/RestoreV2:18*
T0*&
_output_shapes
: @*
use_locking(*
validate_shape(*"
_class
loc:@conv2d_1/kernel
й
save_2/Assign_19Assign
dense/biassave_2/RestoreV2:19*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
о
save_2/Assign_20Assigndense/bias/Adamsave_2/RestoreV2:20*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
░
save_2/Assign_21Assigndense/bias/Adam_1save_2/RestoreV2:21*
T0*
_output_shapes	
:А*
use_locking(*
validate_shape(*
_class
loc:@dense/bias
▓
save_2/Assign_22Assigndense/kernelsave_2/RestoreV2:22*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
╖
save_2/Assign_23Assigndense/kernel/Adamsave_2/RestoreV2:23*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
╣
save_2/Assign_24Assigndense/kernel/Adam_1save_2/RestoreV2:24*
T0* 
_output_shapes
:
АА*
use_locking(*
validate_shape(*
_class
loc:@dense/kernel
ы
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24
1
save_2/restore_allNoOp^save_2/restore_shard "B
save_2/Const:0save_2/Identity:0save_2/restore_all (5 @F8"╢
trainable_variablesЮЫ
7

Variable:0Variable/AssignVariable/read:02Const:0
?
Variable_1:0Variable_1/AssignVariable_1/read:02	Const_1:0
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
X
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
`
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
H
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal:0"
train_op

Adam"═
	variables┐╝
7

Variable:0Variable/AssignVariable/read:02Const:0
?
Variable_1:0Variable_1/AssignVariable_1/read:02	Const_1:0
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
X
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
`
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
H
Variable_2:0Variable_2/AssignVariable_2/read:02truncated_normal:0
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
t
conv2d/kernel/Adam:0conv2d/kernel/Adam/Assignconv2d/kernel/Adam/read:02&conv2d/kernel/Adam/Initializer/zeros:0
|
conv2d/kernel/Adam_1:0conv2d/kernel/Adam_1/Assignconv2d/kernel/Adam_1/read:02(conv2d/kernel/Adam_1/Initializer/zeros:0
l
conv2d/bias/Adam:0conv2d/bias/Adam/Assignconv2d/bias/Adam/read:02$conv2d/bias/Adam/Initializer/zeros:0
t
conv2d/bias/Adam_1:0conv2d/bias/Adam_1/Assignconv2d/bias/Adam_1/read:02&conv2d/bias/Adam_1/Initializer/zeros:0
|
conv2d_1/kernel/Adam:0conv2d_1/kernel/Adam/Assignconv2d_1/kernel/Adam/read:02(conv2d_1/kernel/Adam/Initializer/zeros:0
Д
conv2d_1/kernel/Adam_1:0conv2d_1/kernel/Adam_1/Assignconv2d_1/kernel/Adam_1/read:02*conv2d_1/kernel/Adam_1/Initializer/zeros:0
t
conv2d_1/bias/Adam:0conv2d_1/bias/Adam/Assignconv2d_1/bias/Adam/read:02&conv2d_1/bias/Adam/Initializer/zeros:0
|
conv2d_1/bias/Adam_1:0conv2d_1/bias/Adam_1/Assignconv2d_1/bias/Adam_1/read:02(conv2d_1/bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0
h
dense/bias/Adam:0dense/bias/Adam/Assigndense/bias/Adam/read:02#dense/bias/Adam/Initializer/zeros:0
p
dense/bias/Adam_1:0dense/bias/Adam_1/Assigndense/bias/Adam_1/read:02%dense/bias/Adam_1/Initializer/zeros:0
h
Variable_2/Adam:0Variable_2/Adam/AssignVariable_2/Adam/read:02#Variable_2/Adam/Initializer/zeros:0
p
Variable_2/Adam_1:0Variable_2/Adam_1/AssignVariable_2/Adam_1/read:02%Variable_2/Adam_1/Initializer/zeros:0*Х
serving_defaultБ
/
input&
input:0         2
predict-result 
	Softmax:0         
tensorflow/serving/predict