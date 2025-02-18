import { getServerSession } from "next-auth";
import { type GetServerSidePropsContext } from "next";
import { authConfig } from "./config";

export const auth = (
  req?: GetServerSidePropsContext["req"],
  res?: GetServerSidePropsContext["res"]
) => {
  if (req && res) {
    return getServerSession(req, res, authConfig);
  }
  return null;
};
export { authConfig as config };
