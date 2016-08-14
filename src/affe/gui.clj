(ns affe.gui
  (:require [affe.core :refer :all])
  (:require [uncomplicate.neanderthal.core :refer :all])
  (:import [java.awt.image BufferedImage])
  (:import [javax.swing JComponent JLabel JPanel])
  (:import [java.awt Graphics2D Color GridLayout])
  (:import [java.awt.event ActionEvent ActionListener])
  (:import [mikera.gui Frames]))

(defmacro clamp-colour-value [val]
  `(let [v# (float ~val)]
     (Math/min (float 1.0) (Math/max (float 0.0) v#))))

(defn weight-colour 
  ([^double weight]
    (Color. 
      (clamp-colour-value (Math/tanh (- weight)))
      (clamp-colour-value (Math/tanh weight))
      0.0)))

(defn activation-colour 
  ([^double x]
    (Color. 
      (clamp-colour-value x) 
      (clamp-colour-value (Math/abs x)) 
      (clamp-colour-value (- x)))))

(defn show 
  "Shows a component in a new frame"
  ([com 
    & {:keys [^String title]
       :as options
       :or {title nil}}]
    (Frames/display com (str title))))

(defn default-dimensions
  "Returns the default dimensions for a new frame"
  (^java.awt.Dimension []
    (java.awt.Dimension. 400 300)))

(defn network-graph
     ([nn
       & {:keys [border repaint-speed activation-size line-width max-nodes-displayed] 
          :or {border 20
               repaint-speed 50
               line-width 1
               activation-size 5}}]
       (let [graph (proxy [javax.swing.JComponent java.awt.event.ActionListener] []
           (actionPerformed [^ActionEvent e]
             (.repaint ^JComponent this))
           (paintComponent [^Graphics2D g] 
             (let [border (double border)
                   this ^JComponent this
                   width (double (.getWidth this))
                   height (double (.getHeight this))
                   layers (count (get-layers nn))
                   sizes (vec (map #(ecount %) (get-layers nn)))
                   max-size (reduce max sizes)
                   step (/ (double width) max-size)
                   as (double activation-size)]
               (.setColor g (Color/BLACK))
               (.fillRect g 0 0 width height)
               (.setStroke g (java.awt.BasicStroke. (float line-width))) 
               (dotimes [i (dec layers)]
                 (let [layer (nth (get-layers nn) i)
                       layer-inputs (long (sizes i))
                       layer-outputs (long (sizes (inc i))) 
                       sy (int (+ border (* (- height (* 2 border)) (/ (- layers 0.0 i) layers))))
                       ty (int (+ border (* (- height (* 2 border)) (/ (- layers 1.0 i) layers))))
                       soffset (double border)
                       toffset (double border)
                       sskip (double (/ (- width (* 2 border)) (max 1.0 (dec layer-inputs))))
                       tskip (double (/ (- width (* 2 border)) (max 1.0 (dec layer-outputs))))]
                   (dorun (for [y (sort-by #(hash (* 0.23 %)) (range layer-outputs))] ;; random order of drawing
                     (let [link-count layer-inputs
                           y (int y)
                           tx (int (+ toffset (* tskip y)))]
                       (dotimes [x link-count] 
   	                    (let [ sx (int (+ soffset (* sskip x)))
                               x (int x)]
                             (.setColor g ^Color (weight-colour (entry (nth (get-weights nn) i)  y x)))
   	                        (.drawLine g sx sy tx ty))))))))
               (dotimes [i layers]
                 (let [len (sizes i) 
                       ty (int (+ border (* (- height (* 2 border)) (/ (- layers i) layers))))
                       toffset (double border)
                       tskip (double (/ (- width (* 2 border)) (max 1.0 (dec len))))]
                   (dotimes [y len]
                     (let [activation 0.5
                           tx (int (+ toffset (* tskip y)))]
                       (.setColor g ^Color (activation-colour activation))
                       (.fillRect g (- tx as) (- ty as) (* 2 as) (* 2 as))
                       (.setColor g Color/GRAY)
                       (.drawRect g (- tx as) (- ty as) (* 2 as) (* 2 as)))))))))
             timer (javax.swing.Timer. (int repaint-speed) graph)]
         (.start timer)
         (.setPreferredSize graph (default-dimensions))
         graph)))