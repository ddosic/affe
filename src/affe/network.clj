(ns affe.network
   (:require [affe.protocols :refer :all]
             [uncomplicate.neanderthal.core :refer :all]
             [uncomplicate.commons.core :refer [let-release]]))

(deftype NeuralNetworkVisitor [^affe.protocols.AffeSupportEngine engine]
  Network
  (ff [this network input]
    (feed-forward this network (.wrap-input engine input)))
  NetworkInternals
  (prepare-input [_ input]
    (.wrap-input engine input))
  (prepare-batch [_ input]
    (.wrap-batch engine input))
  (activation-fn [_ x]
    (let-release [cx (copy x) ]
      (.activate-tanh engine cx)))
  (dactivation-fn [_ x]
    (let-release [cx (copy x) ]
      (.deactivate-tanh engine cx)))
  (feed-forward [this network input]
      (let-release [new-activations 
                    (reductions (fn [inp str]
                                   (activation-fn this (mm str inp))) input network)] ;get the new level activations given the inputs
      new-activations
    ))

  (output-deltas [this targets outputs]
    (let-release [dacts (copy (.dactivation-fn this outputs))
            diff  (axpy -1 outputs targets)]
     (.mul engine dacts diff)))
  (update-strengths [_ deltas neurons strengths lrate]
    (axpy strengths (rk lrate deltas neurons)))
  (update-weights [this activations network target learning-rate](
       let-release [strenghts (reverse network) ;get weight vectors between input, hidden levels and output
        layers (reverse activations) ;get layers values
        o-deltas (output-deltas this target (first layers)) ;use the output layer
        h-deltas (->> ;compute the deltas for the hidden layers starting from the output deltas (include the init value: o-deltas)
                  (mapcat #(list [%1 %2]) (rest layers) strenghts) ;list of vectors containing a hidden layer and the weights to the next layer
                  (butlast ,)           ;ignore the input layer
                  (reductions (fn [deltas [neurons strs]] 
                                (let-release [dacts (copy (.dactivation-fn this neurons))
                                              strsT  (trans strs)
                                              prod (mm strsT deltas)]
                                  (.mul engine dacts prod))) 
                                     o-deltas ,) ;produce the deltas for the hidden layers
                  )
        h-deltas-layer-weights (mapcat #(list [%1 %2 %3 learning-rate]) h-deltas (rest layers) strenghts) ;collect values from the three parameter vectors in a single list of vectors
        n-strenghts (map (fn [[deltas neurons strs lr]]
                           (let [neuronsT  (trans neurons)
                                 prod (mm (/ lr (.ncols neurons)) deltas neuronsT)]
                           (axpy strs prod))) h-deltas-layer-weights)] ;produce the new strenghts
       (reverse n-strenghts)))
  )
