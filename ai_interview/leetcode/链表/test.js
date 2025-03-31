function createObj() {
  return {
    name: "whl",
    age: 18,
  };
}

const perA = new createObj();
const s = new Set();
s.add(perA);
s.add(perA);
console.info(s.size);
