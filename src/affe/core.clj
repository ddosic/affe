(ns affe.core
    (:require 
              [uncomplicate.neanderthal.core :refer :all]
              [uncomplicate.fluokitten.core :refer [fmap]]))

(def activation-fn
  "defines the function implemented by a neuron"
  (fn ^double [^double x] (Math/tanh x)))

(def dactivation-fn
  "derivative of activation function"
  (fn ^double [^double y] (- 1.0 (* y y))))

(defn get-weights [network]
  "returns the weights of a network"
  (nth network 1))

(defn get-layers [network]
  "returns the layers of a network, including input and output"
  (first network))

(defn layer-activation [inputs strengths]
   "forward propagate the input of a layer"
   (fmap activation-fn (mv strengths inputs)))

(defn feed-forward [input network]
  "feeds input through the network to the output"
  (let [strenghts (get-weights network) ;get weight vectors between input, hidden levels and output
        new-activations (reductions layer-activation input strenghts)] ;get the new level activations given the inputs
    [(vec (map copy! new-activations (get-layers network)))(get-weights network)])) ; copy new activations into network layer vectors

(defn output-deltas [targets outputs]
  "measures the delta errors for the output layer (Desired value – actual value) and multiplying it by the gradient of the activation function"
  (fmap (fn ^double [^double x ^double y] (* x y)) (fmap dactivation-fn outputs)
     (axpy -1 outputs targets)))

(defn hlayer-deltas [deltas [neurons strengths]]
  "measures the delta errors for the hidden layer based on the output deltas"
  (fmap (fn ^double [^double x ^double y] (* x y)) (fmap dactivation-fn neurons)
     (mv (trans strengths) deltas)))

(defn update-strengths [[deltas neurons strengths lrate]]
  "update the strengths based on the deltas and the learning rate"
  (axpy strengths (rank lrate deltas neurons)))


(defn update-weights [network target learning-rate]
  "updates the weights based on targets and learning rate with back-prop"
  (let [strenghts (reverse (get-weights network)) ;get weight vectors between input, hidden levels and output
        layers (reverse (get-layers network)) ;get layers values
        o-deltas (output-deltas target (first layers)) ;use the output layer
        h-deltas (->> ;compute the deltas for the hidden layers starting from the output deltas (include the init value: o-deltas)
                  (mapcat #(list [%1 %2]) (rest layers) strenghts) ;list of vectors containing a hidden layer and the weights to the next layer
                  (butlast ,)           ;ignore the input layer
                  (reductions hlayer-deltas o-deltas ,) ;produce the deltas for the hidden layers
                  )
        h-deltas-layer-weights (mapcat #(list [%1 %2 %3 learning-rate]) h-deltas (rest layers) strenghts) ;collect values from the three parameter vectors in a single list of vectors
        n-strenghts (map update-strengths h-deltas-layer-weights)] ;produce the new strenghts
    [(get-layers network)(reverse (vec(map copy! n-strenghts strenghts)))])) ; copy new strengths into network matrices

(defn train-network [network input target learning-rate]
  "train network with one set of target data"
  (update-weights (feed-forward input network) target learning-rate))
