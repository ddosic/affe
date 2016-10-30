;;inspired by and reusing some code from Mike Anderson Nurokit demonstration project

(ns affe.gui
  (:require [affe.core :refer :all])
  (:require [uncomplicate.neanderthal.core :refer :all])
  (:import [java.awt.image BufferedImage])
  (:import [javax.swing JComponent JLabel JPanel])
  (:import [java.awt Graphics2D Color GridLayout])
  (:import [java.awt.event ActionEvent ActionListener])
  (:import [java.awt Color Graphics Image])
  (:import [mikera.gui Frames JIcon]))


(declare component)
(declare grid)


(defmacro clamp-colour-value [val]
  `(let [v# (float ~val)]
     (Math/min (float 1.0) (Math/max (float 0.0) v#))))

(defn weight-colour 
  ([^double weight]
    (Color. 
      (if (< weight 0) 1.0 0.0 )
      (if (> weight 0) 1.0 0.0 )
      0.0
      (clamp-colour-value (Math/abs weight))
          )))

(defn activation-colour 
  ([^double x]
    (Color. 
      (clamp-colour-value x) 
      (clamp-colour-value (Math/abs x)) 
      (clamp-colour-value (- x)))))

(defn grid [things]
  (let [n (count things)
        size (int (Math/ceil (Math/sqrt n)))
        grid-layout (GridLayout. 0 size)
        grid (JPanel.)]
    (.setLayout grid grid-layout)
    (doseq [x things]
      (.add grid (component x)))
    grid))

(defn label 
  "Creates a JLabel with the given content"
  (^JLabel [s]
    (let [^String s (str s)
          label (JLabel. s JLabel/CENTER)]
      (.setToolTipText label s)
      label)))

(defn component 
  "Creates a component as appropriate to visualise an object x" 
  (^JComponent [x]
    (cond 
      (instance? JComponent x) x
      (instance? BufferedImage x) (JIcon. ^BufferedImage x)
      (sequential? x) (grid (seq x))
      :else (label x))))

(defn show 
  "Shows a component in a new frame"
  ([com 
    & {:keys [^String title]
       :as options
       :or {title nil}}]  
    (let [com (component com)]
    
    (Frames/display com (str title)))))

 
(defn blank-bitmap [width height]
  (BufferedImage. width height BufferedImage/TYPE_3BYTE_BGR))
 
(defn fill [image color]
  (doto (.getGraphics image)
    (.setColor color)
    (.fillRect 0 0 (.getWidth image) (.getHeight image))))
 
(defn set-pixel [image x y color]
  (.setRGB image x y (.getRGB color)))
 
(defn get-pixel [image x y]
  (Color. (.getRGB image x y)))
 

(defn default-dimensions
  "Returns the default dimensions for a new frame"
  (^java.awt.Dimension []
    (java.awt.Dimension. 400 300)))

(defn network-graph
     ([visitor nn
       & {:keys [border repaint-speed activation-size line-width max-nodes-displayed] 
          :or {border 20
               repaint-speed 50
               line-width 1
               activation-size 2}}]
       (let [graph (proxy [javax.swing.JComponent java.awt.event.ActionListener] []
           (actionPerformed [^ActionEvent e]
             (.repaint ^JComponent this))
           (paintComponent [^Graphics2D g] 
             (let [border (double border)
                   this ^JComponent this
                   width (double (.getWidth this))
                   height (double (.getHeight this))
                   layers (count (.get-layers visitor nn))
                   sizes (vec (map #(ecount %) (.get-layers visitor nn)))
                   max-size (reduce max sizes)
                   step (/ (double width) max-size)
                   as (double activation-size)]
               (.setColor g (Color/WHITE))
               (.fillRect g 0 0 width height)
               (.setStroke g (java.awt.BasicStroke. (float line-width))) 
               (dotimes [i (dec layers)]
                 (let [layer (nth (.get-layers visitor nn) i)
                       layer-inputs (long (sizes i))
                       layer-outputs (long (sizes (inc i))) 
                       sy (int (+ border (* (- height (* 2 border)) (/ (- layers 0.0 i) layers))))
                       ty (int (+ border (* (- height (* 2 border)) (/ (- layers 1.0 i) layers))))
                       soffset (double border)
                       toffset (double border)
                       sskip (double (/ (- width (* 2 border)) (max 1.0 (dec layer-inputs))))
                       tskip (double (/ (- width (* 2 border)) (max 1.0 (dec layer-outputs))))]
                   
                   (dorun (for [y (range layer-outputs)]
                     (let [link-count layer-inputs
                           y (int y)
                           tx (int (+ toffset (* tskip y)))]
                       (dotimes [x link-count] 
   	                    (let [ sx (int (+ soffset (* sskip x)))
                               x (int x)]
                             (.setColor g ^Color (weight-colour (entry (nth (.get-weights visitor nn) i)  y x)))
   	                        (.drawLine g sx sy tx ty))))))
                   ))
               (dotimes [i layers]
                 (let [len (sizes i) 
                       ty (int (+ border (* (- height (* 2 border)) (/ (- layers i) layers))))
                       toffset (double border)
                       tskip (double (/ (- width (* 2 border)) (max 1.0 (dec len))))]
                   (dotimes [y len]
                     (let [activation 0.5
                           tx (int (+ toffset (* tskip y)))]
                       (.setColor g ^Color (activation-colour activation))
                       (.fillOval g (- tx as) (- ty as) (* 2 as) (* 2 as))
                       (.setColor g Color/BLACK)
                       (.drawOval g (- tx as) (- ty as) (* 2 as) (* 2 as)))))))))
             timer (javax.swing.Timer. (int repaint-speed) graph)]
         (.start timer)
         (.setPreferredSize graph (default-dimensions))
         graph)))