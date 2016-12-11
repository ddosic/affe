(ns affe.protocols)

(defprotocol AffeSupportEngine
  (activate-tanh [this x])
  (deactivate-tanh [this x])
  (mul [this x y])
  (gen-strengths [this from to])
  (wrap-input [this input]))

(defprotocol Network
  (ff [this network input]))

(defprotocol NetworkInternals
  (prepare-input [this x])
  (activation-fn [this x])
  (dactivation-fn [this x])
  (feed-forward [this network input])
  (output-deltas [this targets outputs])
  (update-strengths [this deltas neurons strengths lrate])
  (update-weights [this activations network target learning-rate]))