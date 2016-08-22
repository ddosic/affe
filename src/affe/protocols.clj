(ns affe.protocols)

(defprotocol AffeSupportEngine
  (activate-tanh [this x])
  (deactivate-tanh [this x])
  (mul [this x y])
  (gen-layer [this size])
  (gen-strengths [this from to])
  (wrap-input [this input]))

(defprotocol Network
  (get-layers [this network])
  (get-weights [this network])
  (ff [this network input]))

(defprotocol NetworkInternals
  (prepare-input [this x])
  (activation-fn [this x])
  (dactivation-fn [this x])
  (feed-forward [this network input])
  (output-deltas [this targets outputs])
  (update-strengths [this deltas neurons strengths lrate])
  (update-weights [this network target learning-rate]))