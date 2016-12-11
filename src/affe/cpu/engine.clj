(ns affe.cpu.engine
  (:require 
            [affe.protocols :refer :all]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal.native :refer [dge dv]])
)
(deftype AffeNativeEngine []
  AffeSupportEngine
  (activate-tanh [_ x]
    (fmap (fn ^double [^double elem] (Math/tanh elem)) x))
  (deactivate-tanh [_ x]
    (fmap (fn ^double [^double elem] (- 1.0 (* elem elem))) x))
  (mul [_ x y]
    (fmap (fn ^double [^double ix ^double iy] (* ix iy)) x y))
  (gen-strengths [_ from to]
     (let [l (* from to )]
                  (dge to from (vec (repeatedly l (fn [] (rand (/ 1 100))))))))
  (wrap-input [_ input] (dv input))
  )
(defn native-affe-engine
    ([]
       (->AffeNativeEngine)))