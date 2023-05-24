# Some Notation
$out_i \rightarrow$ output of input node $i$ \
$out_h \rightarrow$ output of hidden node $h$ \
$out_o \rightarrow$ output of output node $o$

$net_h \rightarrow$ weighted sum of hidden node $h$  \
$net_o \rightarrow$ weighted sum of output node $o$ 

$\displaystyle w_{i, h} \rightarrow$ weight between node $i$ and node $h$ \
$\displaystyle w_{h, o} \rightarrow$ weight between node $h$ and node $o$

$|out| \rightarrow$ number of nodes in output layer


# Hidden $\rightarrow$ Output
$\displaystyle \frac{\delta E_o}{\delta w_{h, o}} = \frac{\delta E_o}{\delta out_o} \cdot \frac{\delta out_o}{\delta net_o} \cdot \frac{\delta net_o}{\delta w_{h, o}}$

- $\displaystyle \frac{\delta E_o}{\delta out_o} = out_o - target_o$

- $\displaystyle \frac{\delta out_o}{\delta net_o} = out_o (1 - out_o)$

- $\displaystyle \frac{\delta net_o}{\delta w_{h, o}} = out_h$

$\displaystyle \therefore \frac{\delta E_o}{\delta w_{h, o}} = (out_o - target_o) \cdot out_o(1 - out_o) \cdot out_h$




# Input $\rightarrow$ Hidden

$\displaystyle \frac{\delta E_{total}}{w_{i, h}} = \frac{\delta E_{total}}{\delta out_h} \cdot \frac{\delta out_h}{\delta net_h} \cdot \frac{\delta net_h}{\delta w_{i, h}}$

- $\displaystyle \frac{\delta E_{total}}{\delta out_h} = \frac{\delta E_{o1}}{\delta out_h} + \frac{\delta E_{o2}}{\delta out_h} + \dots + \frac{\delta E_{on}}{\delta out_h}$

    - $\displaystyle \frac{\delta E_o}{\delta out_h} = \frac{\delta E_o}{\delta net_o} \cdot \frac{\delta net_o}{\delta out_h}$

        - $\displaystyle \frac{\delta E_o}{\delta net_o} = \frac{\delta E_o}{\delta out_o} \cdot \frac{\delta out_o}{\delta net_o} = (out_o - target_o) \cdot out_o(1 - out_o)$

        - $\displaystyle \frac{\delta net_o}{\delta out_h} = w_{h, o}$

        $\displaystyle \therefore \frac{\delta E_o}{\delta out_h} = (out_o - target_o) \cdot out_o(1 - out_o) \cdot w_{h, o}$

$$
$$

- $\displaystyle \frac{\delta out_h}{\delta net_h} = out_h (1 - out_h)$

- $\displaystyle \frac{\delta net_h}{\delta w_{i, h}} = out_i$

$\displaystyle \therefore \frac{\delta E_{total}}{w_{i, h}} = \sum_{o=0}^{|out|} \bigg [
    \frac{\delta E_{o}}{\delta out_h} \bigg ] \cdot out_h (1 - out_h) \cdot out_i$


$\displaystyle \therefore \frac{\delta E_{total}}{w_{i, h}} = \sum_{o=0}^{|out|} \bigg [(out_o - target_o) \cdot out_o(1 - out_o) \cdot w_{h, o} \bigg ] \cdot out_h (1 - out_h) \cdot out_i$
