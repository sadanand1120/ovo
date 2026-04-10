**Task:** Pick up the book on the table.
```python
table = object_class("table")
book = on_top(table, "book", wholeobj_or_points="wholeobj")
return close_to(book, "floor", wholeobj_or_points="points")
# it forks() the process into branches
```

**Task:** Lift the backpack on the table from the top handle.
```python
table = object_class("table")
backpack = on_top(table, "backpack", wholeobj_or_points="wholeobj")
top_handle = top(backpack).objectify()  # ie, get centroid
return close_to(top_handle, "floor", wholeobj_or_points="points")
```

**Task:** What is written on the front cover of the red notebook on the table?
```python
table = object_class("table")
book = on_top(table, "notebook", wholeobj_or_points="wholeobj")
red_objects = color_appearance("red")
if not book in red_objects:
    return None
in_front_pts = in_front(book, "floor", wholeobj_or_points="points")
below_pts = below(book, "floor", wholeobj_or_points="points")
return union(in_front_pts, below_pts)
# it forks() the process into branches
```

**Task:** Is the side zipper of the backpack on the table broken?
```python
table = object_class("table")
backpack = on_top(table, "backpack", wholeobj_or_points="wholeobj")
side_pts = sides(backpack).objectify()  # ie, get centroid
return close_to(side_pts, "floor", wholeobj_or_points="points")
# it forks() the process into branches
```

**Task:** Pick up the second notebook to the right of the backpack on the table.
```python
table = object_class("table")
backpack = on_top(table, "backpack", wholeobj_or_points="wholeobj")
book1 = on_the_side(backpack, "notebook", wholeobj_or_points="wholeobj")
book2 = on_the_side(book1, "notebook", wholeobj_or_points="wholeobj")
return close_to(book2, "floor", wholeobj_or_points="points")
```