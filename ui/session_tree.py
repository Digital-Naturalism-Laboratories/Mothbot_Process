"""Session tree picker for the Setup tab.

The folders found by a scan are shown as a tree that follows the folder layout
(project › deployment › night, or however deep the folders go) instead of one flat
list. The tree is drawn in the browser from a JSON payload, so collapsing, filtering
and searching don't need a round trip. Ticking boxes writes the selection (and which
rows are collapsed) into a hidden textbox; Python turns that into the same
``selected_paths`` the stage tabs have always read.

Collapsed rows are remembered per chosen folder in ``~/.mothbot/session_tree.json``.
"""

import html
import json
import os
import time

STATE_FILE = os.path.join(os.path.expanduser("~"), ".mothbot", "session_tree.json")
_MAX_REMEMBERED_FOLDERS = 50


# ──────────────────────────────────────────────────────────────
#  Remembered collapse state
# ──────────────────────────────────────────────────────────────

def _state_key(folder_path):
    return os.path.realpath(folder_path) if folder_path else ""


def _read_state_file():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_collapsed(folder_path):
    """Ids of the rows that were collapsed the last time *folder_path* was open."""
    entry = _read_state_file().get(_state_key(folder_path))
    if isinstance(entry, dict) and isinstance(entry.get("collapsed"), list):
        return [str(x) for x in entry["collapsed"]]
    return []


def save_collapsed(folder_path, collapsed):
    key = _state_key(folder_path)
    if not key:
        return
    data = _read_state_file()
    data[key] = {"collapsed": sorted(set(str(x) for x in collapsed)), "savedAt": int(time.time())}
    if len(data) > _MAX_REMEMBERED_FOLDERS:
        newest = sorted(data.items(), key=lambda kv: kv[1].get("savedAt", 0) if isinstance(kv[1], dict) else 0)
        data = dict(newest[-_MAX_REMEMBERED_FOLDERS:])
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass  # Remembering collapsed rows is a convenience; never fail a scan over it.


def parse_tree_state(text):
    """``(selected keys, collapsed ids)`` from the hidden textbox the tree writes."""
    try:
        data = json.loads(text or "{}")
    except ValueError:
        return [], []
    if not isinstance(data, dict):
        return [], []
    selected = [str(x) for x in data.get("selected", []) if isinstance(x, (str, int))]
    collapsed = [str(x) for x in data.get("collapsed", []) if isinstance(x, (str, int))]
    return selected, collapsed


# ──────────────────────────────────────────────────────────────
#  Tree payload
# ──────────────────────────────────────────────────────────────

def make_leaf(key, rel_path, photos, runs, pipeline_flags, is_external, is_legacy, no_project=False):
    """One scanned folder, in the shape the browser tree draws.

    *runs* is ``_detection_run_summaries()``'s result and *pipeline_flags* is
    ``_check_pipeline_status()``'s, both from ``gradio_app``.
    """
    current = runs.get("current") or None
    leaf = {
        "key": key,
        "rel": rel_path,
        "photos": 0 if is_external else int(photos or 0),
        "ext": bool(is_external),
        "legacy": bool(is_legacy),
        # External collections are results without photos: their patches are the detections.
        "d": bool(current) or bool(is_external),
        "c": bool(current and current.get("clustered")),
        "i": bool(current and current.get("identified")),
        "meta": bool(pipeline_flags.get("metadata")),
        "exif": bool(pipeline_flags.get("exif")),
        "pm": bool(current and current.get("pixel_mass")),
        "model": _short_model(current["model"]) if current else None,
        "patches": int(current["patches"]) if current else 0,
        "runPhotos": int(current["photos"]) if current else 0,
        "earlier": [
            {"model": _short_model(r["model"]), "patches": int(r["patches"]), "photos": int(r["photos"])}
            for r in runs.get("archived", [])
        ],
        "human": [int(runs.get("human_patches", 0)), int(runs.get("human_photos", 0))],
        "noProject": bool(no_project),
    }
    return leaf


def _short_model(model_name):
    return str(model_name).removesuffix(".pt").removeprefix("Mothbot_") or "unknown model"


def build_tree(root_name, leaves):
    """Nest *leaves* by their relative folder path.

    Node ids are the relative folder path (``"."`` for the chosen folder itself), so
    they stay the same from one scan to the next and collapsed rows can be remembered.
    """
    root = {"id": "", "name": root_name, "kids": {}, "leaf": None}
    for leaf in leaves:
        rel = (leaf.get("rel") or ".").replace(os.sep, "/").strip("/")
        if rel in ("", "."):
            node = root["kids"].setdefault(".", {"id": ".", "name": root_name, "kids": {}, "leaf": None})
        else:
            node = root
            parts = [p for p in rel.split("/") if p]
            for depth, part in enumerate(parts):
                node_id = "/".join(parts[: depth + 1])
                node = node["kids"].setdefault(part, {"id": node_id, "name": part, "kids": {}, "leaf": None})
        node["leaf"] = leaf

    def finish(node):
        kids = [finish(k) for _, k in sorted(node["kids"].items(), key=lambda kv: kv[0].lower())]
        out = {"id": node["id"], "name": node["name"], "kids": kids}
        if node["leaf"] is not None:
            out["leaf"] = node["leaf"]
        return out

    return finish(root)["kids"]


def tree_html(folder_path, leaves, collapsed=None, selected=None, tree_id="mb-session-tree", state_box="mb-tree-state"):
    """A tree's container: the browser script draws the rows from its payload.

    Several trees can be on the page (Setup, Insert Metadata); each has its own
    *tree_id* and writes its selection into the hidden textbox *state_box*.
    """
    if not leaves:
        return ""
    root_name = os.path.basename(os.path.normpath(folder_path)) or folder_path
    payload = {
        "root": root_name,
        "rootPath": folder_path,
        "nodes": build_tree(root_name, leaves),
        "collapsed": list(collapsed or []),
        "selected": list(selected or []),
        "version": time.time(),
    }
    data = html.escape(json.dumps(payload, separators=(",", ":")), quote=True)
    return (
        f'<div class="mb-session-tree" id="{html.escape(tree_id)}" data-state-box="{html.escape(state_box)}" data-payload="{data}">'
        '<div class="mbt-loading">Loading the folders found…</div></div>'
    )


# ──────────────────────────────────────────────────────────────
#  Browser side (added to the Blocks js / css in gradio_app)
# ──────────────────────────────────────────────────────────────

SESSION_TREE_JS = r"""
            // ── Session tree pickers (ui/session_tree.py) ──────────────────────
            // Each .mb-session-tree on the page keeps its own state, keyed by its id.
            (function() {
                var trees = {};
                var fmt = function(n) { return Number(n || 0).toLocaleString('en-US'); };
                var esc = function(t) {
                    return String(t).replace(/[&<>"]/g, function(c) {
                        return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c];
                    });
                };
                function needsWork(l) { return !(l.d && l.c && l.i); }
                function nextStep(l) {
                    if (!l.d) return 'needs detection';
                    if (!l.c) return 'needs clustering';
                    if (!l.i) return 'needs ID';
                    return 'done';
                }
                function allLeaves(node, out) {
                    if (node.leaf) out.push(node.leaf);
                    (node.kids || []).forEach(function(k) { allLeaves(k, out); });
                    return out;
                }
                function matches(T, l, path) {
                    if (T.filter === 'todo' && !needsWork(l)) return false;
                    if (T.filter === 'done' && needsWork(l)) return false;
                    if (T.query && path.toLowerCase().indexOf(T.query) < 0) return false;
                    return true;
                }
                // Leaves under *node* that pass the filter and search.
                function visibleLeaves(T, node, path, out) {
                    var p = path ? path + '/' + node.name : node.name;
                    if (node.leaf && matches(T, node.leaf, p)) out.push(node.leaf);
                    (node.kids || []).forEach(function(k) { visibleLeaves(T, k, p, out); });
                    return out;
                }
                function checkState(T, list) {
                    var n = 0;
                    list.forEach(function(l) { if (T.sel[l.key]) n++; });
                    return n === 0 ? 'none' : n === list.length ? 'all' : 'some';
                }
                function pips(l) {
                    function pip(on, letter, title, minor) {
                        return '<i class="' + (on ? 'on' : '') + (minor ? ' minor' : '') + '" title="' + title + (on ? ': done' : ': not yet') + '">' + letter + '</i>';
                    }
                    return '<span class="mbt-pips">' + pip(l.d, 'D', 'Detect') + pip(l.c, 'C', 'Cluster') + pip(l.i, 'I', 'ID') +
                        pip(l.meta, 'M', 'Insert Metadata', true) + pip(l.exif, 'E', 'Insert Exif', true) + '</span>';
                }
                function runCell(l) {
                    if (l.ext && !l.model) return '<span class="mbt-run"><em>results only</em></span>';
                    if (!l.model) return '<span class="mbt-run"><em>not detected yet</em></span>';
                    var cov = (l.photos && l.runPhotos < l.photos) ? ' <em>on ' + fmt(l.runPhotos) + '/' + fmt(l.photos) + '</em>' : '';
                    return '<span class="mbt-run">' + esc(l.model) + ': <b>🦋 ' + fmt(l.patches) + '</b>' + cov + '</span>';
                }
                function extras(l) {
                    var bits = [];
                    if (l.noProject) bits.push('no project yet: choose one above');
                    if (l.legacy) bits.push('⚠️ legacy');
                    if (l.ext) bits.push('results only, no photos');
                    (l.earlier || []).forEach(function(r) { bits.push('earlier ' + esc(r.model) + ': 🦋 ' + fmt(r.patches)); });
                    if (l.human && l.human[0]) bits.push('human: 🦋 ' + fmt(l.human[0]) + ' on ' + fmt(l.human[1]) + ' photo' + (l.human[1] === 1 ? '' : 's'));
                    if (l.pm) bits.push('✓ PixMass');
                    return bits.join(' · ');
                }
                function row(T, node, depth, list, isGroup, hasKids, right, small) {
                    var st = checkState(T, list);
                    var open = !T.collapsed[node.id];
                    var caret = hasKids
                        ? '<button type="button" class="mbt-caret" data-toggle="' + esc(node.id) + '" aria-expanded="' + open + '" title="' + (open ? 'Collapse' : 'Expand') + '">▼</button>'
                        : '<span></span>';
                    return '<div class="mbt-row' + (isGroup ? ' mbt-group mbt-depth' + Math.min(depth, 2) : ' mbt-leaf') + (!isGroup && st === 'all' ? ' mbt-selected' : '') + '" style="padding-left:' + (depth * 18 + 4) + 'px">' +
                        caret +
                        '<input type="checkbox" data-check="' + esc(node.id) + '"' + (st === 'all' ? ' checked' : '') + (st === 'some' ? ' data-ind="1"' : '') + ' aria-label="Select ' + esc(node.name) + '">' +
                        '<span class="mbt-name" title="' + esc(node.leaf ? node.leaf.rel : node.id) + '">' + esc(node.name) + (small ? '<small>' + small + '</small>' : '') + '</span>' +
                        '<span class="mbt-cells">' + right + '</span></div>';
                }
                function groupCells(all) {
                    var photos = 0, bugs = 0, todo = 0;
                    all.forEach(function(l) { photos += l.photos; bugs += l.patches; if (needsWork(l)) todo++; });
                    var done = all.length - todo;
                    return '<span>' + (photos ? fmt(photos) : '') + '</span>' +
                        '<span class="mbt-run"><b>' + (bugs ? '🦋 ' + fmt(bugs) : '') + '</b></span>' +
                        '<span><span class="mbt-bar" title="' + done + ' of ' + all.length + ' done"><i style="width:' + Math.round(100 * done / Math.max(1, all.length)) + '%"></i></span></span>' +
                        '<span class="' + (todo ? 'mbt-todo' : 'mbt-done') + '">' + (todo ? todo + ' need' + (todo === 1 ? 's' : '') + ' work' : 'all done') + '</span>';
                }
                function leafCells(l) {
                    var ns = nextStep(l);
                    return '<span>' + (l.photos ? fmt(l.photos) : '') + '</span>' + runCell(l) +
                        '<span>' + pips(l) + '</span><span class="' + (ns === 'done' ? 'mbt-done' : 'mbt-todo') + '">' + ns + '</span>';
                }
                function renderNode(T, node, depth, path) {
                    var p = path ? path + '/' + node.name : node.name;
                    var vis = visibleLeaves(T, node, path, []);
                    if (!vis.length) return '';
                    var kids = node.kids || [];
                    if (!kids.length) {
                        var ex = extras(node.leaf);
                        return row(T, node, depth, vis, false, false, leafCells(node.leaf), ex ? '<span class="mbt-extra">' + ex + '</span>' : '');
                    }
                    var all = allLeaves(node, []);
                    var n = all.length;
                    var groups = kids.filter(function(k) { return k.kids && k.kids.length; }).length;
                    var small = (groups ? groups + ' folder' + (groups === 1 ? '' : 's') + ' · ' : '') + n + ' night' + (n === 1 ? '' : 's');
                    var out = row(T, node, depth, vis, true, true, groupCells(all), small);
                    if (!T.collapsed[node.id]) {
                        // A folder that holds photos itself and also has sub-folders gets its own row.
                        if (node.leaf && matches(T, node.leaf, p)) {
                            out += row(T, { id: node.id + '::self', name: '(photos directly in ' + node.name + ')', leaf: node.leaf }, depth + 1, [node.leaf], false, false, leafCells(node.leaf), extras(node.leaf));
                        }
                        kids.forEach(function(k) { out += renderNode(T, k, depth + 1, p); });
                    }
                    return out;
                }
                function findNode(T, id) {
                    var stack = (T.payload.nodes || []).slice();
                    while (stack.length) {
                        var n = stack.pop();
                        if (n.id === id) return n;
                        (n.kids || []).forEach(function(k) { stack.push(k); });
                    }
                    return null;
                }
                function parentPathOf(T, id) {
                    // The search matches the whole folder path, so rebuild the path above a node.
                    var found = null;
                    (function walk(nodes, path) {
                        nodes.forEach(function(n) {
                            if (found !== null) return;
                            if (n.id === id) { found = path; return; }
                            walk(n.kids || [], path ? path + '/' + n.name : n.name);
                        });
                    })(T.payload.nodes || [], '');
                    return found || '';
                }
                function summary(T) {
                    var sel = T.leaves.filter(function(l) { return T.sel[l.key]; });
                    var photos = 0, undetected = 0;
                    sel.forEach(function(l) { photos += l.photos; if (!l.d) undetected++; });
                    if (!sel.length) return '<span class="mbt-sum">Nothing selected yet. Tick the folders to use.</span>';
                    return '<span class="mbt-sum"><b>' + sel.length + '</b> folder' + (sel.length === 1 ? '' : 's') + ' selected · ' + fmt(photos) + ' photo' + (photos === 1 ? '' : 's') +
                        (undetected ? ' · ' + undetected + ' not detected yet' : '') + '</span>';
                }
                function render(root) {
                    var T = trees[root.id];
                    if (!T || !T.payload) return;
                    var counts = { all: T.leaves.length, todo: 0, done: 0 };
                    T.leaves.forEach(function(l) { if (needsWork(l)) counts.todo++; else counts.done++; });
                    var body = '';
                    (T.payload.nodes || []).forEach(function(n) { body += renderNode(T, n, 0, ''); });
                    if (!body) body = '<div class="mbt-empty">No folders match. Choose “All” or clear the search.</div>';
                    var seg = function(f, label) {
                        return '<button type="button" data-filter="' + f + '" aria-pressed="' + (T.filter === f) + '">' + label + ' <span>' + counts[f] + '</span></button>';
                    };
                    var active = document.activeElement;
                    var focused = active && root.contains(active) && active.classList.contains('mbt-search');
                    var caretPos = focused ? active.selectionStart : null;
                    root.innerHTML =
                        '<div class="mbt-toolbar">' +
                            '<input class="mbt-search" type="search" placeholder="Filter by folder name…" value="' + esc(T.query) + '" aria-label="Filter folders by name">' +
                            '<span class="mbt-seg" role="group" aria-label="Show">' + seg('all', 'All') + seg('todo', 'Needs work') + seg('done', 'Done') + '</span>' +
                            '<span class="mbt-spacer"></span>' +
                            '<button type="button" class="mbt-btn" data-act="todo">Select what needs work</button>' +
                            '<button type="button" class="mbt-btn" data-act="all">Select all</button>' +
                            '<button type="button" class="mbt-btn" data-act="none">Clear</button>' +
                            '<button type="button" class="mbt-btn" data-act="expand">Expand all</button>' +
                            '<button type="button" class="mbt-btn" data-act="fold">Collapse finished</button>' +
                        '</div>' +
                        '<div class="mbt-head"><span></span><span></span><span>' + esc(T.payload.root) + '</span>' +
                            '<span class="mbt-cells"><span>photos</span><span class="mbt-run">current detection run</span><span>stages</span><span>next step</span></span></div>' +
                        '<div class="mbt-rows">' + body + '</div>' +
                        '<div class="mbt-foot">' + summary(T) + '<span class="mbt-legend">' +
                            'D detect · C cluster · I ID · M metadata · E exif &nbsp;·&nbsp; 🦋 insects found by the current run</span></div>';
                    root.querySelectorAll('input[data-ind]').forEach(function(el) { el.indeterminate = true; });
                    if (focused) {
                        var s = root.querySelector('.mbt-search');
                        s.focus();
                        if (caretPos !== null) s.setSelectionRange(caretPos, caretPos);
                    }
                }
                function post(root) {
                    var T = trees[root.id];
                    clearTimeout(T.postTimer);
                    T.postTimer = setTimeout(function() {
                        var id = root.getAttribute('data-state-box');
                        var box = document.querySelector('#' + id + ' textarea, #' + id + ' input');
                        if (!box) return;
                        box.value = JSON.stringify({
                            selected: T.leaves.filter(function(l) { return T.sel[l.key]; }).map(function(l) { return l.key; }),
                            collapsed: Object.keys(T.collapsed).filter(function(k) { return T.collapsed[k]; })
                        });
                        box.dispatchEvent(new Event('input', { bubbles: true }));
                    }, 120);
                }
                function load(root) {
                    var raw = root.getAttribute('data-payload');
                    var T = trees[root.id];
                    if (T && T.raw === raw) {
                        // Same data; Gradio may have redrawn the container (e.g. switching tabs).
                        if (!root.querySelector('.mbt-rows')) render(root);
                        return;
                    }
                    var payload;
                    try { payload = JSON.parse(raw); } catch (e) { return; }
                    // Keep this tree's filter and search across reloads of the same folder.
                    var keep = T && T.payload && T.payload.rootPath === payload.rootPath;
                    T = trees[root.id] = {
                        raw: raw, payload: payload, leaves: [], sel: {}, collapsed: {},
                        filter: keep ? T.filter : 'all', query: keep ? T.query : '', postTimer: null
                    };
                    (payload.nodes || []).forEach(function(n) { allLeaves(n, T.leaves); });
                    (payload.selected || []).forEach(function(k) { T.sel[k] = true; });
                    (payload.collapsed || []).forEach(function(k) { T.collapsed[k] = true; });
                    render(root);
                }
                function treeOf(el) {
                    var root = el && el.closest ? el.closest('.mb-session-tree') : null;
                    return root && trees[root.id] ? root : null;
                }
                document.addEventListener('click', function(e) {
                    var root = treeOf(e.target);
                    if (!root) return;
                    var T = trees[root.id];
                    var t = e.target.closest('[data-toggle]');
                    if (t) {
                        var id = t.getAttribute('data-toggle');
                        T.collapsed[id] = !T.collapsed[id];
                        render(root); post(root); return;
                    }
                    var f = e.target.closest('[data-filter]');
                    if (f) { T.filter = f.getAttribute('data-filter'); render(root); return; }
                    var a = e.target.closest('[data-act]');
                    if (!a) return;
                    var act = a.getAttribute('data-act');
                    if (act === 'todo') T.leaves.forEach(function(l) { T.sel[l.key] = needsWork(l); });
                    if (act === 'all') T.leaves.forEach(function(l) { T.sel[l.key] = true; });
                    if (act === 'none') T.sel = {};
                    if (act === 'expand') T.collapsed = {};
                    if (act === 'fold') {
                        (function walk(nodes) {
                            nodes.forEach(function(n) {
                                if (n.kids && n.kids.length) {
                                    var all = allLeaves(n, []);
                                    if (all.length && !all.some(needsWork)) T.collapsed[n.id] = true;
                                    walk(n.kids);
                                }
                            });
                        })(T.payload.nodes || []);
                    }
                    render(root); post(root);
                });
                document.addEventListener('change', function(e) {
                    var root = treeOf(e.target);
                    if (!root) return;
                    var T = trees[root.id];
                    var id = e.target.getAttribute('data-check');
                    if (id === null) return;
                    var selfRow = id.slice(-6) === '::self';
                    var node = findNode(T, selfRow ? id.slice(0, -6) : id);
                    if (!node) return;
                    var list = selfRow ? [node.leaf] : visibleLeaves(T, node, parentPathOf(T, node.id), []);
                    var to = checkState(T, list) !== 'all';
                    list.forEach(function(l) { T.sel[l.key] = to; });
                    render(root); post(root);
                });
                document.addEventListener('input', function(e) {
                    if (!e.target.classList || !e.target.classList.contains('mbt-search')) return;
                    var root = treeOf(e.target);
                    if (!root) return;
                    trees[root.id].query = e.target.value.trim().toLowerCase();
                    render(root);
                });
                function check() {
                    document.querySelectorAll('.mb-session-tree[data-payload]').forEach(load);
                }
                check();
                new MutationObserver(check).observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-payload'] });
            })();
"""

SESSION_TREE_CSS = """
            /* Session tree picker (ui/session_tree.py) */
            #mb-tree-state, #mb-meta-tree-state, #mb-meta-choices { display: none !important; }
            .mb-session-tree { display: grid; gap: 8px; font-size: 13px; }
            .mb-session-tree .mbt-loading { color: var(--body-text-color-subdued); padding: 8px; }
            .mb-session-tree .mbt-toolbar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
            .mb-session-tree .mbt-spacer { flex: 1; }
            .mb-session-tree .mbt-search {
                font: inherit; padding: 5px 9px; width: 210px; border-radius: 6px;
                border: 1px solid var(--border-color-primary); background: var(--input-background-fill); color: var(--body-text-color);
            }
            .mb-session-tree .mbt-seg { display: inline-flex; border: 1px solid var(--border-color-primary); border-radius: 6px; overflow: hidden; }
            .mb-session-tree .mbt-seg button, .mb-session-tree .mbt-btn {
                height: 30px; margin: 0; line-height: 1; box-sizing: border-box; box-shadow: none; min-width: 0;
            }
            .mb-session-tree .mbt-seg button {
                font: inherit; padding: 5px 10px; border: 0; cursor: pointer;
                background: var(--block-background-fill); color: var(--body-text-color-subdued);
            }
            .mb-session-tree .mbt-seg button + button { border-left: 1px solid var(--border-color-primary); }
            .mb-session-tree .mbt-seg button[aria-pressed="true"] { background: var(--background-fill-secondary); color: var(--body-text-color); font-weight: 700; }
            .mb-session-tree .mbt-btn {
                font: inherit; padding: 5px 10px; border-radius: 6px; cursor: pointer; white-space: nowrap;
                border: 1px solid var(--border-color-primary); background: transparent; color: var(--body-text-color);
            }
            .mb-session-tree .mbt-btn:hover, .mb-session-tree .mbt-seg button:hover { border-color: var(--body-text-color-subdued); }
            .mb-session-tree .mbt-head, .mb-session-tree .mbt-row {
                display: grid; grid-template-columns: 22px 22px minmax(0, 1fr) auto; gap: 4px; align-items: center;
                padding: 4px 10px 4px 4px; border-bottom: 1px solid var(--border-color-primary); font-variant-numeric: tabular-nums;
            }
            .mb-session-tree .mbt-head {
                font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
                color: var(--body-text-color-subdued); border: 1px solid var(--border-color-primary); border-bottom: 0;
                border-radius: 6px 6px 0 0; background: var(--block-background-fill);
            }
            .mb-session-tree .mbt-rows {
                border: 1px solid var(--border-color-primary); border-top: 0; border-radius: 0 0 6px 6px;
                max-height: 520px; overflow-y: auto; background: var(--block-background-fill);
            }
            .mb-session-tree .mbt-row:last-child { border-bottom: 0; }
            .mb-session-tree .mbt-row.mbt-selected { background: color-mix(in srgb, var(--color-accent) 9%, transparent); }
            .mb-session-tree .mbt-group { font-weight: 600; }
            .mb-session-tree .mbt-group.mbt-depth0 { background: var(--background-fill-secondary); font-weight: 700; }
            .mb-session-tree .mbt-caret {
                border: 0; background: none; color: var(--body-text-color-subdued); cursor: pointer;
                font-size: 11px; width: 22px; height: 22px; padding: 0; transition: transform .12s;
            }
            .mb-session-tree .mbt-caret[aria-expanded="false"] { transform: rotate(-90deg); }
            /* Native checkboxes: Gradio's own style draws a white tick that vanishes in the light theme. */
            .mb-session-tree input[type="checkbox"] {
                appearance: auto; -webkit-appearance: checkbox; background: none; background-image: none; border: 0;
                width: 15px; height: 15px; margin: 0; cursor: pointer; accent-color: var(--color-accent);
            }
            .mb-session-tree .mbt-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
            .mb-session-tree .mbt-name small { font-weight: 400; color: var(--body-text-color-subdued); margin-left: 8px; }
            .mb-session-tree .mbt-cells {
                display: grid; grid-template-columns: 76px 230px 108px 118px; gap: 10px; align-items: center;
                font-size: 12px; color: var(--body-text-color-subdued); font-weight: 400;
            }
            .mb-session-tree .mbt-cells > span { text-align: right; white-space: nowrap; }
            .mb-session-tree .mbt-cells > .mbt-run { text-align: left; overflow: hidden; text-overflow: ellipsis; }
            .mb-session-tree .mbt-run b { color: var(--body-text-color); }
            .mb-session-tree .mbt-run em { font-style: normal; }
            .mb-session-tree .mbt-todo { color: var(--color-accent); font-weight: 600; }
            .mb-session-tree .mbt-pips { display: inline-flex; gap: 2px; }
            .mb-session-tree .mbt-pips i {
                font-style: normal; font-size: 10px; font-weight: 700; width: 17px; height: 16px; display: inline-grid; place-items: center;
                border-radius: 3px; border: 1px solid var(--border-color-primary); color: var(--body-text-color-subdued);
            }
            .mb-session-tree .mbt-pips i.on { background: #2e9e5b; border-color: transparent; color: #fff; }
            .mb-session-tree .mbt-pips i.minor { opacity: .7; }
            .mb-session-tree .mbt-pips i.minor.on { background: #6b8f7a; }
            .mb-session-tree .mbt-bar {
                display: inline-block; vertical-align: middle; width: 70px; height: 7px; border-radius: 4px; overflow: hidden;
                border: 1px solid var(--border-color-primary); background: var(--background-fill-secondary);
            }
            .mb-session-tree .mbt-bar i { display: block; height: 100%; background: #2e9e5b; }
            .mb-session-tree .mbt-extra { font-size: 11.5px; }
            .mb-session-tree .mbt-empty { padding: 14px; color: var(--body-text-color-subdued); }
            .mb-session-tree .mbt-foot { display: flex; flex-wrap: wrap; gap: 6px 16px; align-items: baseline; }
            .mb-session-tree .mbt-sum { font-size: 13px; }
            .mb-session-tree .mbt-legend { font-size: 11.5px; color: var(--body-text-color-subdued); margin-left: auto; }
"""
