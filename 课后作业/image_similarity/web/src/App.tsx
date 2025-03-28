import { Image, Layout, Menu, MenuProps } from "antd";
import { Outlet, Routes, Route, Link } from "react-router-dom";

import ImageCompare from "./pages/ImageCompare";
import ImageSearch from "./pages/ImageSearch";
import NotFound from "./pages/NotFound";

import "./App.css";
import { RouterKey } from "./constant";
import { useSelectedKeys } from "./hooks";

const items1: MenuProps["items"] = [
  {
    key: RouterKey.ImageCompare,
    label: <Link to="/image-compare">图片比较</Link>,
  },
  {
    key: RouterKey.ImageSearch,
    label: <Link to="/image-search">图片检索</Link>,
  },
];

const { Header, Content } = Layout;

function App() {
  const selectedKeys = useSelectedKeys();

  return (
    <Layout className="w-screen h-screen">
      <Header className="flex items-center gap-4">
        <div
          style={{ width: 36, height: 36 }}
          className="relative flex items-start justify-center"
        >
          <Image
            src="/bar.webp"
            preview={false}
            wrapperClassName="h-full w-full absolute top-0 left-0"
          />
          <Image
            src="/ball.webp"
            preview={false}
            wrapperClassName="absolute top-0 ball"
          />
        </div>
        <Menu
          theme="dark"
          items={items1}
          mode="horizontal"
          selectedKeys={selectedKeys}
        />
      </Header>
      <Content className="container mx-auto">
        <Routes>
          <Route path="/" element={<Outlet />}>
            <Route index element={<ImageCompare />} />
            <Route path="image-compare" element={<ImageCompare />} />
            <Route path="image-search" element={<ImageSearch />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Content>
    </Layout>
  );
}

export default App;
