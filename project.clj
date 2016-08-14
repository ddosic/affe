(defproject affe "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                  [uncomplicate/neanderthal "0.7.0-SNAPSHOT"]
                  [net.mikera/mikera-gui "0.1.0"]]
  :repl-options {:init-ns affe.core
                 :init (use 'affe.core :reload)}
   :profiles {:dev {:plugins [[lein-midje "3.1.3"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.8.3"]]}}
      )
