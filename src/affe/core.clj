(ns affe.core
    (:require 
              [uncomplicate.neanderthal
               [core :refer :all]
               [native :refer :all]]
              [uncomplicate.fluokitten.core :refer [fmap]]))

(def activation-fn
  "defines the function implemented by a neuron"
  (fn ^double [^double x] (Math/tanh x)))

(def dactivation-fn
  "derivative of activation function"
  (fn ^double [^double y] (- 1.0 (* y y))))

(def range2
  "returns range with interval 2"
  #(take (- %2 %1) (iterate (partial + 2) %1)))

(defn get-weights [network]
  "returns the weights of a network"
  (map first (partition 1 2 (rest network))))

(defn get-layers [network]
  "returns the layers of a network, including input and output"
  (map first (partition 1 2 network)))

(defn layer-activation [inputs strengths]
   "forward propagate the input of a layer"
   (fmap activation-fn (mv strengths inputs)))

(defn gen-strengths [from to]
  "generate random strengths for layer"
  (let [l (* from to )]
    (dge  to from( vec (repeat l 0.01)))))

(defn feed-forward [input network]
  "feeds input through the network to the output"
  (let [strenghts (get-weights network) ;get weight vectors between input, hidden levels and output
        new-activations (reductions layer-activation input strenghts) ;get the new level activations given the inputs
        activations-indexes (cons 0 (map (partial + 2) (range2 0 (count strenghts))))] ;get the indexes corresponding to the position of level activations in network
    (apply (partial assoc network) ; replace values at indexes with replacement values
           (interleave activations-indexes new-activations)))) ; associate positions with replacement values
(defn ff [input network]
  "Feed forward and return output neurons"
  (last (feed-forward (dv input) network)))

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
        n-strenghts (map update-strengths h-deltas-layer-weights) ;produce the new strenghts
        strenghts-indexes (reverse (map inc (range2 0 (count strenghts))))] ;get the indexes corresponding to the position of weights in network
    (apply (partial assoc network) ; replace values at indexes with replacement values
           (interleave strenghts-indexes n-strenghts)))) ; associate positions with replacement values

(defn train-network [network input target learning-rate]
  "train network with one set of target data"
  (update-weights (feed-forward input network) target learning-rate))

(defn train-data [network data learning-rate]
  "train network with a set of data in the form of [[input1 target1] [input2 target2]]"
  (if-let [[input target] (first data)]
    (let [input-block (dv input) 
          target-block (dv target)]
    (recur
     (train-network network input-block target-block learning-rate)
     (rest data)
     learning-rate))
    network))

(defn train-epochs [n network training-data learning-rate]
  "train repeatedly n times over the same tranining data in the form of [[input1 target1] [input2 target2]]  "
  (if (zero? n)
    network
    (recur (dec n)
           (train-data network training-data learning-rate)
           training-data
           learning-rate)))

(defn construct-network
  ([num-in num-hidden num-out]
  "construct a three layer neural network"
  (construct-network num-in num-hidden 0 num-out))
  ([size-in size-hidden num-hidden size-out]
  "construct a N layer neural network"
  (vec (concat
                 [(dv (repeat size-in 0))
                  (gen-strengths size-in size-hidden)
                  (dv (repeat size-hidden 0))]
                 (->>
                  (cons (gen-strengths size-hidden size-hidden) [(dv size-hidden)])
                  (repeat (dec num-hidden))
                  (apply concat))
                 [(gen-strengths size-hidden size-out)
                  (dv (repeat size-out 0))]))))