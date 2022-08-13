# Neural Networks
Neural networks are one approach to machine learning that attempts to deal with the problem of large data dimensionality. The neural network approach uses a fixed number of basis functions - in contrast to methods such as support vector machines that attempt to adapt the number of basis functions - that are themselves parameterized by the model parameters. This is a significant departure from linear regression and logistic regression methods where the models consisted of linear combinations of fixed basis functions, ϕ(x), that dependend only on the input vector, x. In neural networks, the basis functions can now depend on both the model parameters and the input vector and thus take the form ϕ(x|w).

Here we will cover only feed-forward neural networks. One can envision a neural network as a series of layers where each layer has some number of nodes and each node is a function. Each layer represents a single linear regression model. The nodes are connected to each other through inputs accepted from and outputs passed to other nodes. A feed-forward network is one in which these connections do not form any directed cycles. See here for more detail.

As a matter of convention, we will refer the model as a N layer model where N is the number of layers for which adaptive parameters, w, must be determined. Thus for a model consisting of an input layer, one hidden layer, and an output layer, we consider N to be 2 since parameters are determined only for the hidden and output layers.

# Feed-forward Network Functions
We will consider a basic two layer nueral network model, i.e a model that maps inputs to a hidden layer and then to an output layer. We will make the following assumptions
The final output will be a vector Y with K elements, yk, where yk(x,w)=p(C1|x) is the probability that node k is in class C1 and p(C2|x)=1−p(C1|x)
The activation function at a given layer is an arbitrary nonlinear function of a linear combination of the inputs and parameters for that layer
The network is fully connected, i.e. every node at the input layer is connected to every node in the hidden layer and every node in the hidden layer is connected to every node in the output layer
A bias parameter is included at the hidden and output layers
Working from the input layer toward the output layer, we can build this model as follows:

Input Layer
Assume we have an input vector x∈RD. Then the input layer consists of D+1 nodes where the value of the ith node for i=0…D, is 0 if i=0 and xi, i.e. the ith value of x, otherwise.
Hidden Layer
At the hidden layer we construct M nodes where the value of M depends on the specifics of the particular modelling problem. For each node, we define a unit activation, am, for m=1…M as
am=∑Di=0w(1)jixi
where the (1) superscript indicates this weight is for the hidden layer. The output from each node, zm, is then given by the value of a fixed nonlinear function, h, known as the activation function, acting on the unit activation
zm=h(am)=h(∑Di=0w(1)mixi)
Notice that h is the same function for all nodes.
Output Layer
The process at the output layer is essentially the same as at the hidden layer. We construct K nodes, where again the value of K depends on the specific modeling problem. For each node, we again define a unit activation, ak, for k=1…K by
ak=∑Mm=0w(2)kmzm
We again apply a nonlinear activation function, say y, to produce the output
yk=y(ak)
Thus, the entire model can be summarized as a K dimensional output vector Y∈RK where each element yk by
yk(x,w)=y(∑Mm=0w(2)kmh(∑Di=0w(1)mixi))
Generalizations
There are a wide variety of generalizations possible for this model. Some of the more important ones for practical applications include
Addition of hidden layers
Inclusion of skip-layer connections, e.g. a connection from an input node directly to an output node
Sparse network, i.e. not a fully connected network
# Network Training
Here we will consider how to determine the network model parameters. We will use a maximum likelihood approach. The likelihood function for the network is dependent on the type of problem the network models. Of particular importance is the number and type of values generated at the output layer. We consider several cases below.
Regression Single Gaussian Target
Assume that our target variable, t, is Gaussian distributed with an x dependent mean, μ(x) and constant variance σ2=1/β. Our network model is designed to estimate the unknown function for the mean, i.e. y(x,w)≈μ(x) so that the distribution for the target value, t, is modeled by

p(t|x,w)=ND(t|y(x,w),β−1)

where ND(μ,σ2) is the normal distribution with mean μ and variance σ2. Assume the output layer activation function is the identity, i.e. the output is simply the the unit activations, and that we have N i.i.d. observations X=x1,…,x2 and target values t=t1,…,t2, the likelihood function is

p(t|X,w,β)=∏Nn=1p(tn|xn,w,β)

The total error function is defined as the negative logarithm of the likelihood function given by
β2∑Nn=1y(xn,w)−tn2−N2ln(β)+N2ln(2π)
The parameter estimates, w(ML) (ML indicates maximum likelihood) are found by maximizing the equivalent sum-of-squares error function

E(w)=12∑Nn=1y(xn,w)−tn2
Note, due to the nonlinearity of the network model, E(w) is not necessisarily convex, and thus may have local minima and hence it is not possible to know that the global minimum has been obtained. The model parameter, β is found by first finding wML and then solving

1βML=1N∑Nn=1y(xn,w(ML))−tn2
Regression Multiple Gaussian Targets
Now assume that we have K Gaussian distributed target variables, \[t1,…,tK\], each with a mean that is independently conditional on x, i.e. the mean of tk is defined by some function μk(x). Also assume that all K variables share the same variance, σ2=1/β. Assuming the network output layer has K nodes where yk(x,w)≈μk(x) and letting y(x,w)=\[y1(x,w),…,yK(x,w)\], and that we again have N training target values t (t is a K×N matrix of the training values), the conditional distribution of the target training values is given by
p(t|x,w)=ND(t|y(x,w),β−1I)
The parameter estimates, w(ML) are again found by minimizing the sum-of-squares error function, E(w), and the estimate for β is found from

1βML=1NK∑Nn=1||y(xn,w(ML))−tn||2
Binary Classification Single Target
Now assume we have a single target variable, t∈0,1, such that t=1 denotes class C1 and t=0 denotes class C2. Assume that the network has a single output node whos activation function is the logistic sigmoid
y=1/(1+exp(−a))
where a is the output of the hidden layer. We can interpret the network output, y(x,w) as the conditional probability p(C1|x) with p(C2|x)=1−y(x,w) so that the coniditional probability takes a Bernoulli distribution

p(t|x,w)=y(x,w)t\[1−y(x,w)\]1−t
Given a training set with N observations the error function is given by

E(w)=−∑Nn=1tnln(yn)+(1−tn)ln(1−yn)
where yn=y(xn,w). This model assumes all training inputs are correctly labeled.

Binary Classification K Seperate Targets
Assume we have K seperate binary classification target variables (i.e. there are K classification sets each with two classes. The K sets are independent and the input will be assigned to one class from each set). This can be modeled with network having K nodes in the output layer where the activation function of each node is the logistic sigmoid. Assume the class labels are independent, i.e. p(xi∈C1|xj∈C1)=p(xi∈C1) ∀i,j (this is often an invalid assumption in practice), and that we have some training set, t, then the coniditional probability of a single output vector t is
p(t|x,w)=∏Kk=1yk(x,w)tk\[1−yk(x,w)\]1−tk
Given a training input with N values (note that the training set is an K×N matrix) the error function is

E(w=−∑Nn=1∑Kk=1\[tnkln(ynk)+(1−tnk)ln(1−ynk)\]
where $y_{nk} = y_k(\mathbf{x}_n, \mathbf{w}).

Multi-class Classification
Assume we have a K mutually exclusive classes (i.e. the input can only be in one of the K classes). This network is also modelled with K output nodes where each output node represents the probability that the input is in class k, yk(x,w)=p(tk=1|x). The error function is given by
E(w)=−∑Nn=1∑Kk=1tnklnyk(x,w)
where the output activation function yk is the softmax function

yk(x,w)=exp(ak(x,w)∑jexp(aj(x,w))
Error Backpropagation
Most training algorithms involve a two step procedure for minimizing the model parameter dependent error function
Evaluate the derivatives of the error function with respect to the parameters
Use the derivatives to iterate the values of the parameters
In this section, we consider error backpropagation which provides an efficient way to evaluate a feed-forward neural network's error function. Note that this only satisfies step 1 above. It is still necessary to choose an optimization technique in order to use the derivative information provided by error backpropagation to update the parameter estimates as indicated by step 2.

The results presented here are applicable to

An arbitary feed-forward network toplogy
Arbitrary differentiable nonlinear activation functions
The one assumption that is made is that the error function can be expressed as a sum of terms, one for each training data point, tn for n∈1…N, so that

E(w)=∑Nn=1En(w,tn)
For such error functions, the derivative of the error function with respect to $\mathbf{w}) takes the form

▽E(w)=∑Nn=1▽En(w,tn)
Thus we need only consider the evaluation of ▽En(w,tn) which may be used directly for sequential optimization techniques or summed over the training set for batch techniques (see next section). The error backpropagation method can be summarized as follows

Given an input vector, xn, forward propagate through the network using
aj=∑iwjizi
zj=h(aj)
Evaluate δk≡∂En∂ak for the ouptput layer as
δk=yk−tk
Backpropagate the δ's to obtain δj for each hidden unit in the network using
δj=h′(aj)∑kwkjδk
Evaluate the derivatives using
∂En∂wji=δjzi
Parameter Optimization
Given the neural network error function derivative estimate provided by error backpropagation, one can use a variety of numerical optimization techniques to find appropriate parameter estimates. The simplest technique is the method of steepest descent where the new parameter estimate w(τ+1) is given by
w(τ+1)=w(τ)−η▽E(w(τ))
where η>0 is the learning rate. Other more advanced optimization algorithms inlcude conjugate gradients and quasi-Newton methods.

Example: Exlusive or
As a first example, let's consider the classical Exlusive Or (XOR) problem. We will consider the XOR problem involving two inputs x1 and x2 where xi∈0,1. This problem states that the output should be 1 if exactly one of the inputs is 1 and 0 otherwise. Thus this problem has a very simple known input output relationship
x1	x2	Output
0	0	0
1	0	1
0	1	1
1	1	0
While this may be a trivial example, it illustrates all of the features of an arbitary feed-forward network while remaining simple enough to understand everything that is happening in the network. The XOR network is typically presented as having 2 input nodes, 2 hidden layer nodes, and one output nodes, for example see here. Such an arrangement however requires that the nodes in the hidden layer use a thresholding scheme whereby the output of the node is 0 or 1 depending on the sum of the input being greater than some fixed threshold value. However, such a network would violoate our model assumptions for training the network. Specifically, we could not use error backpropagation because the node activation functions would be step functions which are not differentiable. To overcome this, we will the hyperbolic tangent, tanh, as the hidden layer activation function and add a third node to the hidden layer representing a bias term. Our output layer will have a single node. We will interpret the out values as being 0 (or false) for output values less than 0 and 1 (or true) for output values greater than 0. The figure below provides a graphical representation of the network. Note, the parameters w and a are distinct for each layer, e.g. w11 on the edge between x1 and h(a1) is not the same w11 on the edge from h(a1) to σ(a1).

