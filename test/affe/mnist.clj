(ns affe.mnist
  (:require [clojure.java.io :as io]
            [midje.sweet :refer :all]
            [affe.core :refer :all]
            [affe.network :refer :all]
            [affe.cpu.engine :refer :all]
            [affe.trainer :refer :all]
            [affe.gui :refer :all]
            [uncomplicate.clojurecl.core  :refer [*context* *command-queue* with-default]]
            [uncomplicate.neanderthal
             [opencl :refer [with-default-engine]]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.native :refer [sge sv]])
  (:import [java.io DataInputStream File FileInputStream BufferedInputStream]))

(set! *unchecked-math* true) 

(def mnist-path "data/mnist")

(defn ^DataInputStream get-data-stream [name]
  (DataInputStream. (io/input-stream (io/resource (str mnist-path name)))))


(def CASE-COUNT 60000)

(def SIZE 28)

(def TEST-CASE-COUNT 10000)

(def ub-to-double-factor (double (/ 1.0 255.0)))

(def data-store (future
	(with-open [^DataInputStream data-input-stream (get-data-stream "/train-images.idx3-ubyte")]
	   (let [datavector (atom [])
	       ]
		  (if (not= (.readInt data-input-stream) 2051)
		    (throw (Error. "Wrong magic number")))
		  (if (not= (.readInt data-input-stream) 60000)
		    (throw (Error. "Unexpected image count")))  
		  (if (not= (.readInt data-input-stream) SIZE)
		    (throw (Error. "Unexpected row count")))  
		  (if (not= (.readInt data-input-stream) SIZE)
		    (throw (Error. "Unexpected column count")))  
		  (dotimes [i CASE-COUNT]
		    (let [darray (float-array (* SIZE SIZE))]
			    (dotimes [y SIZE]
			      (dotimes [x SIZE]
			        (aset-float 
		            darray 
		            (+ x (* y SIZE)) 
		            (* ub-to-double-factor (.readUnsignedByte data-input-stream)))))
		      (swap! datavector conj (vec darray))))
	   @datavector))))

(def label-store (future
	(with-open [^DataInputStream data-input-stream (get-data-stream "/train-labels.idx1-ubyte")]
   (let [labelvector (atom [])]
	  (if (not= (.readInt data-input-stream) 2049)
	    (throw (Error. "Wrong magic number")))
	  (if (not= (.readInt data-input-stream) 60000)
	    (throw (Error. "Unexpected image count")))  
	  (dotimes [i CASE-COUNT]
	    (do
	      (swap! labelvector conj (long (.readUnsignedByte data-input-stream)))))
    @labelvector))))

(def test-data-store (future
	(with-open [^DataInputStream data-input-stream (get-data-stream "/t10k-images.idx3-ubyte") ]
    (let [datavector (atom [])]
		  (if (not= (.readInt data-input-stream) 2051)
		    (throw (Error. "Wrong magic number")))
		  (if (not= (.readInt data-input-stream) TEST-CASE-COUNT)
		    (throw (Error. "Unexpected image count")))  
		  (if (not= (.readInt data-input-stream) SIZE)
		    (throw (Error. "Unexpected row count")))  
		  (if (not= (.readInt data-input-stream) SIZE)
		    (throw (Error. "Unexpected column count")))  
		  (dotimes [i TEST-CASE-COUNT]
		    (let [darray (float-array (* SIZE SIZE))]
			    (dotimes [y SIZE]
			      (dotimes [x SIZE]
			        (aset-float 
		            darray 
		            (+ x (* y SIZE)) 
		            (* ub-to-double-factor (.readUnsignedByte data-input-stream)))))
		      (swap! datavector conj (vec darray))))
	   @datavector))))

(def test-label-store (future
	(with-open [^DataInputStream data-input-stream (get-data-stream "/t10k-labels.idx1-ubyte")]
    (let [labelvector (atom [])]
		  (if (not= (.readInt data-input-stream) 2049)
		    (throw (Error. "Wrong magic number")))
		  (if (not= (.readInt data-input-stream) TEST-CASE-COUNT)
		    (throw (Error. "Unexpected image count")))  
		  (dotimes [i TEST-CASE-COUNT]
		    (do
		      (swap! labelvector conj (long (.readUnsignedByte data-input-stream)))))
	    @labelvector))))

(defn vectorize-digit [digit]
  (assoc (vec (repeat 10 0)) digit 1))

(defn devectorize-digit [vect]
  (first (apply max-key second (map-indexed vector vect))))

(defn execute [visitor net input] (last (first (.ff  visitor net input))))

(defn get-precision [count visitor network test-label-store test-data-store]
  (double (/ (reduce + 
                     (map (fn [l r] (if (= l r) 1 0))
                          (take count test-label-store)
                          (map devectorize-digit (map (partial execute visitor network)(take count test-data-store)))))
             count)))

(facts
 "Hopefully no crash here."
  
 (def naf (native-affe-engine))
 (def visitor (->NeuralNetworkVisitor naf))
 (def testNet (construct-network naf 784 160 10))
 (def data  (take 20000 @data-store))
 (def labels (take 20000 @label-store))
 ;(def trained (train visitor testNet 20 (vec (map (fn [input response] [input (vectorize-digit response)]) data labels )) 0.1))
 
 ;(show (network-graph visitor trained))
 
 ;(get-precision 1000 visitor trained @label-store @data-store)
 ;(get-precision 1000 visitor trained @test-label-store @test-data-store)
 
 
 
)
