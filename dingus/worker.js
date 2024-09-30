const PARSER_PACKAGE = "./wheel/lrparsers-0.7.9-py3-none-any.whl"


// Load the whole pyodide thingy.
importScripts("pyodide/pyodide.js");

const dingus_module = {
  post_grammar_status: function (message) {
    console.log("Grammar Status:", message);
    postMessage({kind: "grammar_status", status: "loading", message});
  },

  post_grammar_loaded: function () {
    console.log("Grammar Loaded");
    postMessage({kind: "grammar_status", status: "ok", message: "Grammar loaded"});
  },

  post_grammar_error: function(error) {
    console.log("Grammar Error:", error);
    postMessage({kind:"grammar_status", status: "error", message: error});
  },
};

async function setup_python() {
  console.log("Loading pyodide....");
  const pyodide = await loadPyodide({
    packages: ["micropip"],
  });
  pyodide.setStdout({ batched: (msg) => console.log(msg) }); // TODO: I know this is an option above.

  // TODO: Do I actually want micropip? Probably not?
  console.log("Installing parser package...");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(PARSER_PACKAGE);

  pyodide.registerJsModule("dingus", dingus_module);

  pyodide.runPython(`
import dingus
import parser
import pyodide.code

GRAMMAR_GLOBALS = {}

def eval_grammar(code):
  global GRAMMAR_GLOBALS
  try:
    dingus.post_grammar_status("Evaluating grammar...")
    pyodide.code.eval_code(code, globals=GRAMMAR_GLOBALS)
    dingus.post_grammar_loaded()
  except Exception as e:
    dingus.post_grammar_error(f"{e}")
`);

  console.log("Loaded!");
  self.pyodide = pyodide;
  return pyodide;
}
const pyodide_promise = setup_python();


async function load_grammar_module(code) {
  const pyodide = self.pyodide;

  console.log("Running...");

  const my_fn = pyodide.globals.get("eval_grammar");
  my_fn(code);
  my_fn.destroy();
}

self.onmessage = async function(event) {
  await pyodide_promise;

  try {
    const { kind, data } = event.data;
    if (kind === "grammar_module") {
      try {
        await load_grammar_module(data);
      } catch (e) {
        console.log("INTERNAL ERROR:", e.message);
        postMessage({error: e.message});
      }
    }
  } catch (wtf) {
    console.log("WTF?");
    console.log(wtf);
  }
};
