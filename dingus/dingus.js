// TODO: Abstract/Factor
let pending_grammar = null;
let next_grammar = null;

let pending_document = null;
let next_document = null;

const STATUS = document.getElementById("status-line");

const DOC_CHAINS = {};
function chain_document_submit(kind, editor, on_success) {
  let pending = null;
  let next = null;

  function do_submit(document) {
    if (pending) {
      console.log("Old still pending, parking it...");
      next = document;
    } else {
      pending = document;
      worker.postMessage({kind, data: document});
      console.log("Document submitted");
    }
  }

  function on_result(message) {
    pending = null;
    if (next) {
      pending = next;
      next = null;

      worker.postMessage({kind, data: document});
      console.log("Posted another document");
    }
    if (message.kind === "ok" && on_success) {
      on_success(message);
    }
  }
  DOC_CHAINS[kind] = on_result;

  let change_timer_id = null;
  editor.doc.on("change", () => {
    clearTimeout(change_timer_id);
    change_timer_id = setTimeout(() => {
      change_timer_id = null;
      do_submit(editor.doc.getValue());
    }, 100);
  });
}

const worker = new Worker('worker.js');
worker.onmessage = (e) => {
  const message = e.data;

  const chain = DOC_CHAINS[message.kind];
  if (chain) {
    chain(message);
  }

  STATUS.innerText = message.message;
};


function setup_editors() {
  const grammar_editor = CodeMirror.fromTextArea(
    document.getElementById("grammar"),
    {
      lineNumbers: true,
      mode: "python",
      value: "from parser import Grammar\n\nclass MyGrammar(Grammar):\n    pass\n",
    },
  );
  chain_document_submit("grammar", grammar_editor);

  const input_editor = CodeMirror.fromTextArea(
    document.getElementById("input"),
    {
      lineNumbers: true,
    },
  );
  chain_document_submit("input", input_editor);
}

setup_editors();
