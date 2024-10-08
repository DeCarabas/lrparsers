const STATUS = document.getElementById("status-line");
const ERRORS = document.getElementById("error-root");
const TREE = document.getElementById("tree-root");

const INITIAL_STATE = {
  worker: null,
  status: "Initializing...",
  tree: null,
  errors: [],
  grammar: {
    last: null,
    pending: null,
    next: null,
  },
  input: {
    last: null,
    pending: null,
    next: null,
  },
};


/*
 * Render a tree `node` into the DOM.
 */
function render_tree_node(node, input_editor) {
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
  if (node.start == node.end) {
    node_label.classList.add("parsed-error-node");
  }
  tree_div.appendChild(node_label);


  if (node.kind === "tree") {
    tree_div.classList.add("parsed-tree");
    for (const child of node.children) {
      tree_div.appendChild(render_tree_node(child, input_editor));
    }
  } else {
    tree_div.classList.add("parsed-token");
  }

  return tree_div;
}


/*
 * Render the state out into the DOM, the bindings to which we have already
 * established. Fully side-effecting, no shadow DOM here.
 */
function render_state(state, input_editor) {
  STATUS.innerText = state.status;

  ERRORS.innerText = state.errors.join("\n");

  TREE.innerHTML = "";
  if (state.tree) {
    TREE.appendChild(render_tree_node(state.tree, input_editor));
  }
}

/*
 * Post a changed document out to the worker, if the worker can take it,
 * otherwise just queue it for submission.
 */
function post_document(worker, kind, state, document) {
  if (window.localStorage) {
    window.localStorage.setItem(kind, document);
  }

  let new_state = {...state};
  if (new_state.pending) {
    new_state.next = document;
  } else {
    new_state.pending = document;
    new_state.next = null;
    worker.postMessage({kind, data: document});
  }
  return new_state;
}

/*
 * Handle a document submission by rotating in the next document to be
 * submitted. (Documents flow from next -> pending -> last.)
 */
function rotate_document(worker, kind, state) {
  let new_state = {...state, last: state.pending, pending: null};
  if (new_state.next) {
    new_state.pending = new_state.next;
    new_state.next = null;
    worker.postMessage({kind, data: new_state.pending});
  }
  return new_state;
}

/*
 * Update the state given the message and return a new state.
 *
 * This can be side-effecting, in that it might also post messages to the worker
 * (to rotate documents and whatnot). (Maybe we should do something about that?)
 */
function update(state, message) {
  let new_state = {...state};
  if (message.message) {
    new_state.status = message.message;
  }

  if (message.kind === "grammar") {
    if (message.status === "changed") {
      new_state.grammar = post_document(
        new_state.worker,
        "grammar",
        new_state.grammar,
        message.data,
      );
    } else if (message.status === "ok" || message.status === "error") {
      new_state.grammar = rotate_document(new_state.worker, "grammar", new_state.grammar);

      if (message.status === "ok") {
        // Re-submit the input, using whatever the latest document state was.
        new_state.input = post_document(
          new_state.worker,
          "input",
          new_state.input,
          new_state.input.next || new_state.input.pending || new_state.input.last,
        );
      }

      if (message.status === "error") {
        // Record the errors.
        new_state.errors = [];
      }
    }
  }

  if (message.kind === "input") {
    if (message.status === "changed") {
      new_state.input = post_document(
        new_state.worker,
        "input",
        new_state.input,
        message.data,
      );
    } else if (message.status === "ok" || message.status === "error") {
      new_state.input = rotate_document(new_state.worker, "input", new_state.input);

      if (message.status === "ok") {
        new_state.tree = message.tree;
        new_state.errors = message.errors;
      }

      if (message.status === "error") {
        new_state.errors = message.errors;
      }
    }
  }

  //console.log(state, message, new_state);
  return new_state;
}

/*
 * Construct a new "message handler" by wrapping up the state into an update/
 * render loop.
 */
function message_handler(worker, input_editor) {
  let state = {...INITIAL_STATE, worker};
  render_state(state, input_editor);

  return (message) => {
    state = update(state, message);
    render_state(state, input_editor);
  };
}

// And we can set up the codemirror editors to publish their changes.
function setup_editors() {
  function setup_editor(kind, editor, handler) {
    let change_timer_id = null;
    editor.doc.on("change", () => {
      clearTimeout(change_timer_id);
      change_timer_id = setTimeout(() => {
        change_timer_id = null;
        handler({
          kind,
          status: "changed",
          data: editor.doc.getValue(),
        });
      }, 100);
    });

    if (window.localStorage) {
      const value = window.localStorage.getItem(kind);
      if (value) {
        editor.doc.setValue(value);
      }
    }
  }

  // Construct the editors and worker and stuff but don't wire the handler
  // yet...
  const worker = new Worker('worker.js');
  const grammar_editor = CodeMirror.fromTextArea(
    document.getElementById("grammar"),
    {
      lineNumbers: true,
      mode: "python",
    },
  );
  const input_editor = CodeMirror.fromTextArea(
    document.getElementById("input"),
    {
      lineNumbers: true,
    },
  );

  // ...now we can construct the handler with what it needs....
  const handler = message_handler(worker, input_editor);

  // ...and finally hook the event sources to the handler.
  worker.onmessage = (e) => {
    const message = e.data;
    handler(message);
  };

  setup_editor("grammar", grammar_editor, handler);
  setup_editor("input", input_editor, handler);
}

setup_editors();
