import hashlib, itertools, json

def get_format_fn(val) -> str:
    return """(coldef) => { 
            coldef._number_format = Intl.NumberFormat(navigator.language, { minimumFractionDigits : %u, maximumFractionDigits: %u });
            coldef.fmtfn = function(val) {               
                return this._number_format.format(val);
            };
            coldef.fmtfn.bind(coldef);
        };""" % (val, val)

def create_treeview_grouping(rows: list[dict[str, any]], groupby: list[str], ordering: dict[dict[str, int]]|None = None) -> list[dict[str, any]]:
    sort_keys = {}
    for grpcol in groupby:
        if ordering is not None and grpcol in ordering:
            sort_keys[grpcol] = lambda row: ordering[grpcol][row[grpcol]]
        else:
            sort_keys[grpcol] = lambda row: row[grpcol]

    def recursive_fn(parent: dict[str, any], hash: object, rows: list[dict[str, any]], groupby: list[str], depth: int):
        rcount = 0
        gcount = 0
        ret_group_list = []
        grpcol = groupby.pop(0)
        for k, g in itertools.groupby(rows, key=lambda row: row[grpcol]):
            grouprows = list(g)
            start = rows.index(grouprows[0])
            end = start + len(grouprows)
            hh = hash.copy()
            hh.update(json.dumps(k).encode())
            group = dict(id=hh.hexdigest(), title=str(k), colid=grpcol, depth=depth, groupcount=0, rowcount=end-start, rows=(start, end))
            ret_group_list.append(group)
            rcount += end - start
            gcount += 1
            if len(groupby):
                ret_group_list += recursive_fn(group, hh, rows, groupby, depth+1)
        parent["rowcount"] -= rcount
        parent["groupcount"] = gcount
        return ret_group_list
    
    for grpcol in reversed(groupby):
        rows.sort(key=sort_keys[grpcol])
    
    h = hashlib.sha256()
    h.update(b"root")
    rowcount = len(rows)
    root = dict(id=h.hexdigest(), title="All", depth=0, groupcount=0, rowcount=rowcount, rows=(0, rowcount))
    ret = [ root ]
    ret += recursive_fn(root, h, rows, groupby, 1)
    return ret
