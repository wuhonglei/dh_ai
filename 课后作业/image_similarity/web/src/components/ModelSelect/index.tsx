import { Select } from "antd";
import { options } from "./constant";

import classNames from "classnames";

type Props = {
  value?: string;
  className?: string;
  onChange?: (value: string) => void;
};

export default function ModelSelect(props: Props) {
  const { value, className, onChange } = props;

  return (
    <Select
      value={value}
      options={options}
      onChange={onChange}
      className={classNames("w-32", className)}
    />
  );
}
