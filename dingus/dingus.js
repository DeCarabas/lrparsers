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

  function do_submit() {
    const document = editor.doc.getValue();
    if (window.localStorage) {
      window.localStorage.setItem(kind, document);
    }

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

    if (message.status === "ok" && on_success) {
      on_success(message);
    }
  }
  DOC_CHAINS[kind] = on_result;

  let change_timer_id = null;
  editor.doc.on("change", () => {
    clearTimeout(change_timer_id);
    change_timer_id = setTimeout(() => {
      change_timer_id = null;
      do_submit();
    }, 100);
  });

  if (window.localStorage) {
    const value = window.localStorage.getItem(kind);
    if (value) {
      editor.doc.setValue(value);
    }
  }

  return do_submit;
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

let grammar_editor = null;
let input_editor = null;

function render_parse_results(message) {
  console.log("WHAT?");

  function render_tree_node(parent, node) {
    const tree_div = document.createElement("div");
    tree_div.classList.add("parsed-node");

    const node_label = document.createElement("a");
    node_label.innerText = node.name;
    node_label.setAttribute("href", "#");
    node_label.onclick = () => {
      const doc = input_editor.doc;
      doc.setSelection(
        doc.posFromIndex(node.start),
        doc.posFromIndex(node.end),
        {scroll: true},
      );
    };
    tree_div.appendChild(node_label);

    if (node.kind === "tree") {
      tree_div.classList.add("parsed-tree");
      for (const child of node.children) {
        render_tree_node(tree_div, child);
      }
    } else {
      tree_div.classList.add("parsed-token");
    }
    parent.appendChild(tree_div);
  }

  const root = document.getElementById("output-root");
  root.innerHTML = "";
  render_tree_node(root, message.tree);
}

function setup_editors() {
  grammar_editor = CodeMirror.fromTextArea(
    document.getElementById("grammar"),
    {
      lineNumbers: true,
      mode: "python",
    },
  );

  input_editor = CodeMirror.fromTextArea(
    document.getElementById("input"),
    {
      lineNumbers: true,
    },
  );

  const submit_input = chain_document_submit(
    "input", input_editor, render_parse_results);

  chain_document_submit(
    "grammar", grammar_editor, submit_input);
}

setup_editors();
