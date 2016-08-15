(ns affe.neanderthal.native
(:require [affe.core :refer :all])
(:require [uncomplicate.neanderthal
             [core :refer :all]
             [native :refer :all]]))

(defn gen-strengths [from to]
  "generate random strengths for layer"
  (let [l (* from to )]
    (dge  to from( vec (repeat l 0.01)))))

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
  ([size-in size-hidden size-out]
  "construct a three layer neural network"
  (construct-network size-in size-hidden 1 size-out))
  ([size-in size-hidden num-hidden size-out]
  "construct a N layer neural network"
    [(vec (concat
                    [(dv size-in)]
                    (->>
                     [(dv size-hidden)]
                     (repeat  num-hidden)
                     (apply concat))
                     [(dv size-out)]))
     (vec (concat
                    [(gen-strengths size-in size-hidden)]
                    (->>
                      [(gen-strengths size-hidden size-hidden)]
                      (repeat (dec num-hidden))
                      (apply concat))
                    [(gen-strengths size-hidden size-out)]))
     ]
   ))

(defn ff [input network]
  "Feed forward and return output neurons"
  (last (get-layers (feed-forward (dv input) network))))
