(ns affe.opencl.engine
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release let-release with-release wrap-int]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [count-work-groups enq-reduce
                              enq-read-int enq-read-double]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [block :refer :all]
             [core :refer [dim mrows ncols transfer transfer!]]
             [native :refer [dv dge]]]
            [uncomplicate.neanderthal.opencl :refer [clv clge with-default-engine]]
            [uncomplicate.neanderthal.opencl.clblock :refer :all]
            [affe.protocols :refer :all])
)
(deftype AffeCLEngine [ctx cqueue prog]
  Releaseable
  (release [_]
    (release prog))
  AffeSupportEngine
  (activate-tanh [_ x]
        (with-release [tanh-kernel (kernel prog "act_tanh")]
          (set-args! tanh-kernel 0 (buffer x)(wrap-int (.offset x))
                   (wrap-int (.stride x)))
          (enq-nd! cqueue tanh-kernel (work-size-1d (dim x))))
        x)
  (deactivate-tanh [_ x]
        (with-release [detanh-kernel (kernel prog "deact_tanh")]
          (set-args! detanh-kernel 0 (buffer x)(wrap-int (.offset x))
                   (wrap-int (.stride x)))
          (enq-nd! cqueue detanh-kernel (work-size-1d (dim x))))
        x)
  (mul [_ x y]
        (with-release [mul-kernel (kernel prog "mul")]
          (set-args! mul-kernel 0 (buffer x)(wrap-int (.offset x))
                   (wrap-int (.stride x))(buffer y)(wrap-int (.offset y))
                   (wrap-int (.stride y)))
          (enq-nd! cqueue mul-kernel (work-size-1d (dim x))))
        x)
  (gen-strengths [_ from to]
    (let [l (* from to )]
      (clge  to from( vec (repeat l 0.01)))))
  (wrap-input [_ input] (clv input))
  )

  (defn cl-affe-engine
    ([ctx cqueue]
     (let-release [prog (build-program!
                         (program-with-source ctx [(slurp (io/resource "affe/opencl/kernel/support_functions.cl"))])
                         "-DREAL=float" nil)]
       (->AffeCLEngine ctx cqueue prog))))