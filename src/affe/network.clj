(ns affe.network
   (:require [affe.protocols :refer :all]
             [uncomplicate.neanderthal.core :refer :all]
             [uncomplicate.commons.core :refer [let-release release-seq]]))

(deftype NeuralNetworkVisitor [^affe.protocols.AffeSupportEngine engine]
  Network
  (get-layers [_ network]
    (first network))
  (get-weights [_ network]
    (nth network 1))
  (ff [this network input]
    (feed-forward this network (.wrap-input engine input)))
  NetworkInternals
  (prepare-input [_ input]
    (.wrap-input engine input))
  (activation-fn [_ x]
    (let-release [cx (copy x) ]
      (.activate-tanh engine cx)))
  (dactivation-fn [_ x]
    (let-release [cx (copy x) ]
      (.deactivate-tanh engine cx)))
  (feed-forward [this network input]
      (let-release [strenghts (get-weights this network) ;get weight vectors between input, hidden levels and output
        new-activations (reductions (fn [inp str](activation-fn this (mv str inp))) input strenghts)] ;get the new level activations given the inputs
    (release-seq (get-layers this network))
     [new-activations strenghts] 
    ))
  (output-deltas [this targets outputs]
    (let-release [dacts (copy (.dactivation-fn this outputs))]
     (.mul engine dacts (axpy -1 outputs targets))))
  (update-strengths [_ deltas neurons strengths lrate]
    (axpy strengths (rank lrate deltas neurons)))
  (update-weights [this network target learning-rate](
       let-release [strenghts (reverse (get-weights this network)) ;get weight vectors between input, hidden levels and output
        layers (reverse (get-layers this network)) ;get layers values
        o-deltas (output-deltas this target (first layers)) ;use the output layer
        h-deltas (->> ;compute the deltas for the hidden layers starting from the output deltas (include the init value: o-deltas)
                  (mapcat #(list [%1 %2]) (rest layers) strenghts) ;list of vectors containing a hidden layer and the weights to the next layer
                  (butlast ,)           ;ignore the input layer
                  (reductions (fn [deltas [neurons strs]] 
                                (let-release [dacts (copy (.dactivation-fn this neurons))] 
                                  (.mul engine dacts (mv (trans strs) deltas)))) 
                                     o-deltas ,) ;produce the deltas for the hidden layers
                  )
        h-deltas-layer-weights (mapcat #(list [%1 %2 %3 learning-rate]) h-deltas (rest layers) strenghts) ;collect values from the three parameter vectors in a single list of vectors
        n-strenghts (map (fn [[deltas neurons strs lr]] (axpy strs (rank lr deltas neurons))) h-deltas-layer-weights)] ;produce the new strenghts
       (release-seq strenghts)
       [(.get-layers this network) (reverse n-strenghts)]))
  )
