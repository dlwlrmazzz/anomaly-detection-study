Auto-Encoding Variational Bayes 라는 paper에 대해 리뷰하고자 한다.

## Introduction
저자들은 continuous latent variables나 parameters with intractable posterior를 가진 model을 학습하거나 추론시에 approximation을 잘하는 방법을 찾고 싶어 했다. 그래서 이 논문을 통해 Auto-Encoding VB algorithm을 제시하는 것이다.

## Problem setting
dataset $\textbf{X} = \{ \textbf{x}^{(i)} \}_{i=1}^{N}$이고 각 sample은 iid continuous or discrete varable이다. 이러한 data가 만들어지는 과정에 있어서 latent(unobserved) continuous random variable $\textbf{z}$가 있다고 가정하자. 그 과정은 그렇다면 두 가지로 생각할 수 있는데
- a value $\textbf{z}^{(i)}$가 prior distribution $p(\textbf{z};\theta)$에서 생성된다.
- a value $\textbf{x}^{(i)}$가 conditional distribution $p(\textbf{x}\|\textbf{z};\theta)$에서 생성된다.

또한 위의 두 prior와 likelihood는 parametric하며 $\theta, \textbf{z}$에 대해 미분가능 하다고 한다.

이러한 경우에 있어서 *Intractability, large dataset*의 경우에도 문제를 해결할 수 있는 solution을 찾아야한다.
- Intractability : marginal likelihood $p(\textbf{x})=\int p(\textbf{z};\theta) p(\textbf{x}\|\textbf{z};\theta)d\textbf{z}$가 intractable한 경우를 의미한다. 이는 posterior $p(\textbf{z}\|\textbf{x})=p(\textbf{x}\|\textbf{z};\theta)p(\textbf{z};\theta)/p(\textbf{x};\theta)$ 또한 구할 수 없게 된다.
- large dataset : Sampling기반의 방법들은 너무 느리다.

그래서 저자들은 해결하고 싶은 3가지를 제시한다.
- Efficient approximate ML or MAP estimation for the parameters $\theta$
  - parameter를 잘 구한다면 이전에는 몰랐던 generation process에 대해 파악할 수 있다.
- Efficient approximate posterior inference of the latent variable $\textbf{z}$ given an observed value $\textbf{x}$ for a choice of parameters $\theta$
  - latent variable에 대한 파악을 통해 representation task가 가능하다.
- Efficient approximate marginal inference of the variable $\textbf{x}$
  - $\textbf{x}$에 대한 이해 필요한 task, 예를 들어 vision분야에서 image denoising, super-resolution 등의 task를 할 수 있게 된다.

이를 위해 저자들은 recognition model $q(\textbf{z}\|\textbf{x};\phi)$를 이용한다. 이는 intractable true posterior $p(\textbf{z}\|\textbf{x};\theta)$에 대한 approximation이다. generative model에 해당하는 $p(\textbf{x}\|\textbf{z};\theta)$와 함께 parameter를 학습하게 된다.

## Variational bound

이전에 variational inference를 통해 아래의 과정을 이해할 수 있다. log marginal likelihood는 $\sum \log p(\textbf{x}^{(i)})$이고 각 data point마다 살펴보자. 

$$\log p(\textbf{x}^{(i)};\theta)=D_{KL}[q(\textbf{z}|\textbf{x};\phi) || p(\textbf{z}|\textbf{x};\theta)]+L(\theta,\phi,;\textbf{x}^{(i)})$$

$$\log p(\textbf{x}^{(i)};\theta) \ge L(\theta,\phi,;\textbf{x}^{(i)})=E_{q(\textbf{z}|\textbf{x};\phi)}[-\log q(\textbf{z}|\textbf{x};\phi)+\log p(\textbf{x},\text{z};\theta)] \\= -D_{KL}[q(\textbf{z}|\textbf{x};\phi) || p(\textbf{z};\theta)]+E_{q(\textbf{z}|\textbf{x};\phi)}[\log p(\textbf{x}^{(i)}|\textbf{z};\theta)]$$

결국 우리는 Lower bound를 최대화해야 한다. 이를 위해 Lower bound는 variational parameter $\phi$와 generative parameter $\theta$에 대해 미분하여 optimize한다. 바로 구할 수 없기 때문에 이를 위해서 lower bound에 대한 estimator가 필요하고 이는 sampling에 기반한다. 아래에서 살펴보자.

## SGVB estimator and AEVB algorithm
위에서 lower bound에 대한 식을 2개 살펴보았다. 각각에 대한 estimator를 알아볼 것이다. 후자가 variance가 더 작기에 우리는 후자를 사용한다. (sampling을 덜 하기 때문) 후자의 경우 KL-divergence term이 analytically 구할 수 있기 때문이다. (appendix 참고)

$$\tilde{L1}(\theta,\phi;\textbf{x}^{(i)})=\frac{1}{L}\sum_{l=1}^L [ \log p(\textbf{x}^{(i)},\textbf{z}^{(i,l)};\theta)-\log q(\textbf{z}^{(i,l)}|\textbf{x}^{(i)};\phi)]\\ \text{where}\; \textbf{z}^{(i,l)}=g(\epsilon^{(i,l)},\textbf{x}^{(i)})\;\;\text{and}\;\epsilon^{(l)}\sim p(\epsilon)$$

$$\tilde{L2}(\theta,\phi;\textbf{x}^{(i)})=-D_{KL}[q(\textbf{z}|\textbf{x};\phi) || p(\textbf{z};\theta)]+\frac{1}{L}\sum_{l=1}^L[\log p(\textbf{x}^{(i)}|\textbf{z}^{(i,l)};\theta)]\\ \text{where}\; \textbf{z}^{(i,l)}=g(\epsilon^{(i,l)},\textbf{x}^{(i)})\;\;\text{and}\;\epsilon^{(l)}\sim p(\epsilon)$$

위의 식을 이용하여 우리는 $M$개의 data point로 이루어진 minibatch에 기반하여 estimator를 구할 수 있다.

$$L(\theta, \phi; \textbf{X})\approx\tilde{L}^M (\theta, \phi; \textbf{X}^M)=\frac{N}{M}\sum_{i=1}^M \tilde{L}(\theta, \phi; \textbf{x}^{(i)}) $$

## Reparameterization trick




## Appendix