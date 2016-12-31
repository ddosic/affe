(ns affe.core-opencl-engine-test
  (:require [midje.sweet :refer :all]
            [affe.core :refer :all]
            [affe.protocols :refer :all]
            [affe.network :refer :all]
            [affe.opencl.engine :refer :all]
            [affe.cpu.engine :refer :all]
            [affe.gui :refer :all]
            [uncomplicate.clojurecl.core  :refer [*context* *command-queue* with-default finish!]]
            [uncomplicate.neanderthal
             [core :refer [transfer]]
             [native :refer [dge]]
             [opencl :refer [clge with-default-engine]]]
            ))
(with-default
  (with-default-engine
    (facts
       "Hopefully no crash here."
       
       (def engine (cl-affe-engine *context* *command-queue*))
       (def engine-cpu (native-affe-engine))
       (def deact (.deactivate-tanh engine (clge 3 3 (vec (range 9)))))
       (def act (.activate-tanh engine (clge 3 3 (vec (range 9)))))
       (def mult (.mul engine (clge 3 3 (vec (range 9)))(clge 3 3 (vec (range 9)))))
       (def host-deact (transfer deact))
       (def host-act  (transfer act))
       (def host-mult  (transfer mult))
       (println "deact " host-deact)
       (println "act " host-act)
       (println "mul " host-mult)
       (println "deact cpu " (.deactivate-tanh engine-cpu (dge 3 3 (vec (range 9)))))
       (println "act cpu " (.activate-tanh engine-cpu (dge 3 3 (vec (range 9)))))
       (println "mul cpu " (.mul engine-cpu (dge 3 3 (vec (range 9)))(dge 3 3 (vec (range 9)))))
     )))

