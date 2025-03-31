from typing import TypedDict, Literal


class CategoryItem(TypedDict):
    id: int
    name: str
    display_name: str
    parent_id: int
    has_children: bool
    has_active_children: bool
    children: list['CategoryItem']


class LeafCategoryItem(TypedDict):
    id: int
    name: str
    display_name: str
    parent_id: int
    has_children: Literal[False]
    has_active_children: Literal[False]
    children: list[None]
